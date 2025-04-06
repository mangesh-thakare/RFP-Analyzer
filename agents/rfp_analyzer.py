import os
import sys
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
from pydantic import BaseModel, Field
import tempfile
import shutil
import PyPDF2
import json
from docx import Document
from dotenv import load_dotenv
from datetime import datetime

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()


# Import the workflows
from agents.eligibility_workflow import EligibilityWorkflow
from agents.submission_checklist import RFPAnalyzer
from agents.contract_risk_agent import analyze_contract_risks
from utils.text_extraction import extract_text_from_pdf

# Define the company data model
class CompanyData(BaseModel):
    company_name: str = Field(..., description="Name of the company")
    capabilities: list[str] = Field(default_factory=list, description="List of company capabilities")
    certifications: list[str] = Field(default_factory=list, description="List of company certifications")
    experience: Dict = Field(default_factory=dict, description="Company experience details")
    insurance: Dict = Field(default_factory=dict, description="Insurance information")
    key_personnel: list[Dict] = Field(default_factory=list, description="Key personnel details")
    past_performance: list[Dict] = Field(default_factory=list, description="Past performance details")
    technical_capabilities: list[Dict] = Field(default_factory=list, description="Technical capabilities")

app = FastAPI(title="RFP Analysis API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize workflows
eligibility_workflow = EligibilityWorkflow()
submission_analyzer = RFPAnalyzer()

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file"""
    try:
        doc = Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return "\n".join(text)
    except Exception as e:
        raise ValueError(f"Error extracting text from DOCX: {str(e)}")

@app.post("/upload-company-data/")
async def upload_company_data(document: UploadFile = File(...)):
    """Upload and store company document in ChromaDB"""
    try:
        # Check file extension
        file_extension = document.filename.lower().split('.')[-1]
        if file_extension not in ['pdf', 'docx']:
            return JSONResponse(
                status_code=400,
                content={"error": "Only PDF and DOCX files are accepted for company documents"}
            )
        
        # Save the document temporarily
        temp_path = f"temp_{document.filename}"
        try:
            with open(temp_path, "wb") as buffer:
                document.file.seek(0)
                shutil.copyfileobj(document.file, buffer)
            
            # Extract text based on file type
            if file_extension == 'pdf':
                # Verify PDF is readable
                with open(temp_path, 'rb') as test_file:
                    PyPDF2.PdfReader(test_file)
                document_text = extract_text_from_pdf(temp_path)
            else:  # docx
                document_text = extract_text_from_docx(temp_path)
            
            if not document_text:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Could not extract text from document"}
                )
            
            # Create company data structure from document
            company_dict = {
                "company_name": "Company from Document",
                "capabilities": [],
                "certifications": [],
                "experience": {},
                "insurance": {},
                "key_personnel": [],
                "past_performance": [],
                "technical_capabilities": [],
                "company_document": {
                    "filename": document.filename,
                    "content": document_text
                }
            }
            
            # Store document in ChromaDB
            result = submission_analyzer.load_company_data(company_dict)
            if "error" in result:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Failed to store company document: {result['error']}"}
                )
            
            return {
                "status": "success",
                "message": "Successfully stored company document",
                "timestamp": datetime.now().isoformat(),
                "filename": document.filename
            }
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing company document: {str(e)}"
        )

@app.post("/analyze-rfp/")
async def analyze_rfp(file: UploadFile = File(...)):
    """Analyze RFP document for eligibility, submission requirements, and contract risks in parallel"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Check if company data exists
    if submission_analyzer.vector_stores["company"] is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Company data not found. Please upload company data first using /upload-company-data/ endpoint."}
        )

    temp_path = None
    try:
        # Save the uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            file.file.seek(0)
            shutil.copyfileobj(file.file, buffer)
        
        # Verify PDF is readable
        try:
            with open(temp_path, 'rb') as test_file:
                PyPDF2.PdfReader(test_file)
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid PDF file: {str(e)}"
            )

        # Load the document for submission analysis
        submission_analyzer.load_rfp_document(temp_path)

        # Run all three analyses in parallel
        eligibility_task = eligibility_workflow.process_rfp(file)
        submission_task = asyncio.create_task(
            asyncio.to_thread(submission_analyzer.extract_submission_checklist, "RFP_1")
        )
        contract_risk_task = analyze_contract_risks(extract_text_from_pdf(temp_path))

        # Wait for all tasks to complete
        eligibility_result, submission_result, contract_risk_result = await asyncio.gather(
            eligibility_task, submission_task, contract_risk_task
        )

        # Check for errors in any result
        if "error" in eligibility_result:
            return JSONResponse(
                status_code=400,
                content={"error": f"Eligibility analysis failed: {eligibility_result['error']}"}
            )

        if "error" in submission_result:
            return JSONResponse(
                status_code=400,
                content={"error": f"Submission checklist generation failed: {submission_result['error']}"}
            )

        if "error" in contract_risk_result:
            return JSONResponse(
                status_code=400,
                content={"error": f"Contract risk analysis failed: {contract_risk_result['error']}"}
            )

        # Combine results
        response = {
            "rfp_id": "RFP_1",
            "timestamp": submission_result["timestamp"],
            "eligibility_analysis": eligibility_result,
            "submission_checklist": submission_result["checklist"],
            "contract_risk_analysis": contract_risk_result
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing RFP: {str(e)}"
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

@app.post("/analyze-eligibility/")
async def analyze_eligibility(file: UploadFile = File(...)):
    """Analyze RFP for eligibility requirements only"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    return await eligibility_workflow.process_rfp(file)

@app.post("/analyze-submission/")
async def analyze_submission(file: UploadFile = File(...)):
    """Analyze RFP for submission requirements only"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    temp_path = None
    try:
        # Save the uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            file.file.seek(0)
            shutil.copyfileobj(file.file, buffer)
        
        # Verify PDF is readable
        try:
            with open(temp_path, 'rb') as test_file:
                PyPDF2.PdfReader(test_file)
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid PDF file: {str(e)}"
            )

        # Load the document and analyze
        submission_analyzer.load_rfp_document(temp_path)
        result = submission_analyzer.extract_submission_checklist("RFP_1")

        if "error" in result:
            return JSONResponse(
                status_code=400,
                content={"error": f"Submission checklist generation failed: {result['error']}"}
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing RFP: {str(e)}"
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

@app.post("/analyze-contract-risks/")
async def analyze_contract_risks_endpoint(file: UploadFile = File(...)):
    """Analyze RFP for contract risks only"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    temp_path = None
    try:
        # Save the uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            file.file.seek(0)
            shutil.copyfileobj(file.file, buffer)
        
        # Verify PDF is readable
        try:
            with open(temp_path, 'rb') as test_file:
                PyPDF2.PdfReader(test_file)
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid PDF file: {str(e)}"
            )

        # Extract text and analyze
        contract_text = extract_text_from_pdf(temp_path)
        if not contract_text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        result = await analyze_contract_risks(contract_text)
        return {
            "status": "success",
            "filename": file.filename,
            "analysis": result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing RFP: {str(e)}"
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

@app.get("/")
async def root():
    return {
        "message": "RFP Analysis API",
        "endpoints": {
            "/upload-company-data/": "Upload and store company data",
            "/analyze-rfp/": "Analyze RFP for eligibility, submission requirements, and contract risks",
            "/analyze-eligibility/": "Analyze RFP for eligibility requirements only",
            "/analyze-submission/": "Analyze RFP for submission requirements only",
            "/analyze-contract-risks/": "Analyze RFP for contract risks only"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 