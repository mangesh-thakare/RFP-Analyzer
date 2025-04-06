import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from typing import List, Dict, Optional
from dotenv import load_dotenv
import PyPDF2
import json
import time
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils.text_extraction import extract_text_from_pdf
from tempfile import NamedTemporaryFile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RFP Analyzer API",
    description="API for analyzing RFP documents and extracting requirements",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def estimate_tokens(text: str) -> int:
    """Estimate number of tokens in text"""
    return len(text.split()) * 1.3

def chunk_text(text: str, max_tokens: int = 1500) -> List[str]:
    """
    Split text into chunks based on estimated token count
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)
        
        if paragraph_tokens > max_tokens:
            sentences = [s.strip() for s in re.split(r'[.!?]+', paragraph) if s.strip()]
            for sentence in sentences:
                sentence_tokens = estimate_tokens(sentence)
                if current_tokens + sentence_tokens <= max_tokens:
                    current_chunk += sentence + ". "
                    current_tokens += sentence_tokens
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
                    current_tokens = sentence_tokens
        else:
            if current_tokens + paragraph_tokens <= max_tokens:
                current_chunk += paragraph + "\n\n"
                current_tokens += paragraph_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
                current_tokens = paragraph_tokens
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

async def process_chunk(groq: ChatGroq, chunk: str, chunk_num: int) -> dict:
    """Process a single chunk with error handling"""
    prompt = """Extract ONLY EXPLICITLY STATED requirements from this RFP text section. 
DO NOT infer or add requirements that are not directly mentioned in the text.

Rules for extraction:
1. Only extract requirements that are EXPLICITLY stated in the text
2. Do not include generic federal regulations (like FAR, CFR) unless specifically mentioned as a requirement
3. Do not include Executive Orders (E.O.) unless specifically mentioned as a requirement
4. For each requirement, it must be clearly stated as mandatory using words like "must", "shall", "required", "mandatory", "will provide"
5. Ignore standard legal references unless explicitly required for compliance
6. Only include certifications that are specifically asked for in the RFP

For each requirement, look for:
1. Service Category: The EXACT type of service being requested
2. Mandatory Requirements: Only those explicitly stated as required
3. Optional Requirements: Only those explicitly marked as preferred/optional
4. Experience: Only explicit minimum years required
5. Insurance: Only specific coverage types and amounts mentioned
6. Key Personnel: Only positions explicitly required

Return ONLY a JSON object with this structure:
{
    "mandatory_requirements": {
        "registrations_and_licenses": ["only explicitly required licenses"],
        "certifications": ["only explicitly required certifications"]
    },
    "optional_requirements": {
        "registrations_and_licenses": ["only explicitly optional registrations"],
        "certifications": ["only explicitly optional certifications"]
    },
    "experience_requirements": ["only explicit year requirements"],
    "insurance_requirements": {
        "Workers Compensation": "only if explicitly required with limits",
        "General Liability": "only if explicitly required with limits"
    },
    "service_category": "exact service type mentioned",
    "key_personnel": ["only explicitly required positions"]
}

Text section:
""" + chunk

    try:
        response = groq.invoke(prompt)
        time.sleep(0.5)  # Reduced sleep time since we're parallel processing
        
        json_str = response.content.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].strip()
        
        chunk_results = json.loads(json_str)
        print(f"Successfully processed chunk {chunk_num}")
        return chunk_results
    except Exception as e:
        print(f"Warning: Error processing chunk {chunk_num}: {str(e)}")
        return {}

def merge_results(results: List[dict]) -> dict:
    """Merge results from multiple chunks"""
    combined_results = {
        "mandatory_requirements": {
            "registrations_and_licenses": set(),
            "certifications": set()
        },
        "optional_requirements": {
            "registrations_and_licenses": set(),
            "certifications": set()
        },
        "experience_requirements": set(),
        "insurance_requirements": {},
        "service_category": "",
        "key_personnel": set()
    }
    
    for chunk_result in results:
        if not chunk_result:
            continue
            
        # Update mandatory requirements
        if "mandatory_requirements" in chunk_result:
            if isinstance(chunk_result["mandatory_requirements"].get("registrations_and_licenses"), list):
                valid_items = [
                    item for item in chunk_result["mandatory_requirements"]["registrations_and_licenses"]
                    if isinstance(item, str) and len(item) > 5 and 
                    not any(x in item.lower() for x in ["required license", "required registration", "specific", "license 1", "registration 1"])
                ]
                combined_results["mandatory_requirements"]["registrations_and_licenses"].update(valid_items)
            
            if isinstance(chunk_result["mandatory_requirements"].get("certifications"), list):
                valid_items = [
                    item for item in chunk_result["mandatory_requirements"]["certifications"]
                    if isinstance(item, str) and len(item) > 5 and 
                    not any(x in item.lower() for x in ["required certification", "certification 1", "specific"])
                ]
                combined_results["mandatory_requirements"]["certifications"].update(valid_items)
        
        # Update optional requirements
        if "optional_requirements" in chunk_result:
            if isinstance(chunk_result["optional_requirements"].get("registrations_and_licenses"), list):
                valid_items = [
                    item for item in chunk_result["optional_requirements"]["registrations_and_licenses"]
                    if isinstance(item, str) and len(item) > 5 and 
                    not any(x in item.lower() for x in ["preferred license", "preferred registration", "specific", "license 1", "registration 1"])
                ]
                combined_results["optional_requirements"]["registrations_and_licenses"].update(valid_items)
            
            if isinstance(chunk_result["optional_requirements"].get("certifications"), list):
                valid_items = [
                    item for item in chunk_result["optional_requirements"]["certifications"]
                    if isinstance(item, str) and len(item) > 5 and 
                    not any(x in item.lower() for x in ["preferred certification", "certification 1", "specific"])
                ]
                combined_results["optional_requirements"]["certifications"].update(valid_items)
        
        # Update experience requirements
        if isinstance(chunk_result.get("experience_requirements"), list):
            for exp in chunk_result["experience_requirements"]:
                if isinstance(exp, str):
                    years = re.findall(r'\d+', exp)
                    if years:
                        combined_results["experience_requirements"].update(years)
        
        # Update insurance requirements
        if isinstance(chunk_result.get("insurance_requirements"), dict):
            for key, value in chunk_result["insurance_requirements"].items():
                if isinstance(key, str) and not "insurance_type" in key.lower():
                    combined_results["insurance_requirements"][key] = value
        
        # Update service category
        if isinstance(chunk_result.get("service_category"), str):
            service_cat = chunk_result["service_category"].strip()
            if "staffing" in service_cat.lower() or "temporary" in service_cat.lower():
                combined_results["service_category"] = "Temporary Staffing Services"
        
        # Update key personnel
        if isinstance(chunk_result.get("key_personnel"), list):
            valid_items = [
                item for item in chunk_result["key_personnel"]
                if isinstance(item, str) and len(item) > 2 and 
                not any(x in item.lower() for x in ["specific position", "position 1"])
            ]
            combined_results["key_personnel"].update(valid_items)
    
    return {
        "mandatory_requirements": {
            "registrations_and_licenses": sorted(list(combined_results["mandatory_requirements"]["registrations_and_licenses"])),
            "certifications": sorted(list(combined_results["mandatory_requirements"]["certifications"]))
        },
        "optional_requirements": {
            "registrations_and_licenses": sorted(list(combined_results["optional_requirements"]["registrations_and_licenses"])),
            "certifications": sorted(list(combined_results["optional_requirements"]["certifications"]))
        },
        "experience_requirements": min([int(year) for year in combined_results["experience_requirements"]] or [0]),
        "insurance_requirements": combined_results["insurance_requirements"],
        "service_category": combined_results["service_category"] or "Temporary Staffing Services",
        "key_personnel": sorted(list(combined_results["key_personnel"]))
    }

async def extract_rfp_requirements(rfp_text: str) -> dict:
    """
    Extract key requirements from RFP documents using parallel processing
    """
    groq = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.2-3b-preview"
    )
    
    chunks = chunk_text(rfp_text)
    
    # Create tasks for parallel processing
    tasks = []
    for i, chunk in enumerate(chunks, 1):
        task = process_chunk(groq, chunk, i)
        tasks.append(task)
    
    # Process chunks in parallel
    results = await asyncio.gather(*tasks)
    
    # Merge results from all chunks
    return merge_results(results)

@app.post("/analyze-rfp/")
async def analyze_rfp_endpoint(file: UploadFile = File(...)):
    """
    Analyze an RFP document and extract requirements
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    try:
        with NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            rfp_text = extract_text_from_pdf(temp_file_path)
            if not rfp_text:
                raise HTTPException(status_code=400, detail="Could not extract text from PDF")
            
            requirements = await extract_rfp_requirements(rfp_text)
            
            return {
                "status": "success",
                "filename": file.filename,
                "requirements": requirements
            }
            
        finally:
            os.unlink(temp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing RFP: {str(e)}")

@app.get("/")
async def root():
    """
    Root endpoint - provides API information
    """
    return {
        "message": "Welcome to RFP Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze-rfp/": "POST - Upload and analyze RFP document",
            "/": "GET - This information"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)