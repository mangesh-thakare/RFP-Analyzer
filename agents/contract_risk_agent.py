import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from typing import List, Dict, Optional
from dotenv import load_dotenv
import json
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils.text_extraction import extract_text_from_pdf
from tempfile import NamedTemporaryFile
import asyncio

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Contract Risk Analyzer API",
    description="API for analyzing contract clauses and identifying potential risks",
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

def chunk_text(text: str, max_tokens: int = 1500) -> List[str]:
    """
    Split text into chunks based on estimated token count
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for paragraph in paragraphs:
        paragraph_tokens = len(paragraph) // 4  # Rough estimation
        
        if paragraph_tokens > max_tokens:
            sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
            for sentence in sentences:
                sentence_tokens = len(sentence) // 4
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
    prompt = """Analyze this contract text section for potential risks and biased clauses that could put ConsultAdd at a disadvantage.

Focus on identifying:
1. Unilateral termination rights or unfair termination clauses
2. Unreasonable payment terms or delays
3. Excessive liability or indemnification requirements
4. Unfair intellectual property rights
5. Unreasonable performance guarantees
6. One-sided modification rights
7. Unfair dispute resolution clauses
8. Unreasonable confidentiality obligations
9. Unfair non-compete clauses
10. Unreasonable warranty requirements

For each identified risk, provide:
1. The specific clause or language
2. Why it's problematic
3. Suggested modifications to make it more balanced

Return ONLY a JSON object with this structure:
{
    "risks": [
        {
            "clause": "exact problematic clause text",
            "risk_type": "type of risk (e.g., termination, payment, liability)",
            "risk_level": "high/medium/low",
            "explanation": "why this is problematic",
            "suggested_modification": "how to make it more balanced"
        }
    ]
}

Text section:
""" + chunk

    try:
        response = groq.invoke(prompt)
        time.sleep(0.5)  # Rate limiting
        
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
        return {"risks": []}

def merge_results(results: List[dict]) -> dict:
    """Merge results from multiple chunks"""
    all_risks = []
    risk_types = {}
    
    for chunk_result in results:
        if not chunk_result or "risks" not in chunk_result:
            continue
            
        for risk in chunk_result["risks"]:
            # Deduplicate similar risks
            is_duplicate = False
            for existing_risk in all_risks:
                if (risk["clause"].lower() in existing_risk["clause"].lower() or 
                    existing_risk["clause"].lower() in risk["clause"].lower()):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_risks.append(risk)
                risk_type = risk["risk_type"]
                risk_types[risk_type] = risk_types.get(risk_type, 0) + 1
    
    # Sort risks by risk level (high -> medium -> low)
    risk_level_order = {"high": 0, "medium": 1, "low": 2}
    all_risks.sort(key=lambda x: risk_level_order[x["risk_level"]])
    
    return {
        "risks": all_risks,
        "risk_summary": {
            "total_risks": len(all_risks),
            "risk_type_distribution": risk_types,
            "risk_level_distribution": {
                "high": len([r for r in all_risks if r["risk_level"] == "high"]),
                "medium": len([r for r in all_risks if r["risk_level"] == "medium"]),
                "low": len([r for r in all_risks if r["risk_level"] == "low"])
            }
        }
    }

async def analyze_contract_risks(contract_text: str) -> dict:
    """
    Analyze contract text for potential risks and biased clauses
    """
    groq = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.2-3b-preview"
    )
    
    chunks = chunk_text(contract_text)
    
    # Create tasks for parallel processing
    tasks = []
    for i, chunk in enumerate(chunks, 1):
        task = process_chunk(groq, chunk, i)
        tasks.append(task)
    
    # Process chunks in parallel
    results = await asyncio.gather(*tasks)
    
    # Merge results from all chunks
    return merge_results(results)

@app.post("/analyze-contract-risks/")
async def analyze_contract_risks_endpoint(file: UploadFile = File(...)):
    """
    Analyze a contract document for potential risks and biased clauses
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    try:
        with NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            contract_text = extract_text_from_pdf(temp_file_path)
            if not contract_text:
                raise HTTPException(status_code=400, detail="Could not extract text from PDF")
            
            analysis = await analyze_contract_risks(contract_text)
            
            return {
                "status": "success",
                "filename": file.filename,
                "analysis": analysis
            }
            
        finally:
            os.unlink(temp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing contract: {str(e)}")

@app.get("/")
async def root():
    """
    Root endpoint - provides API information
    """
    return {
        "message": "Welcome to Contract Risk Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze-contract-risks/": "POST - Upload and analyze contract for risks",
            "/": "GET - This information"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001) 