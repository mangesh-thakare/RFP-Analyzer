import os
import json
import re
import traceback
from typing import Dict, List
from datetime import datetime
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from IPython.display import display, Markdown

# Load environment variables from .env file
load_dotenv()

class RFPAnalyzer:
    def __init__(self):
        # Get API key from environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.llm = ChatGroq(
            model_name="llama-3.2-3b-preview",
            temperature=0.1,  # Slightly higher temperature for better creativity
            groq_api_key=groq_api_key
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize vector stores
        self.vector_stores = {
            "rfp": None,
            "company": None
        }
        
        # Create or load company vector store
        self.company_store_path = "./chroma_db/company"
        if os.path.exists(self.company_store_path):
            self.vector_stores["company"] = Chroma(
                persist_directory=self.company_store_path,
                embedding_function=self.embeddings
            )

    def load_company_data(self, company_data: Dict) -> Dict:
        """Load and store company data in ChromaDB"""
        try:
            # Convert company data to text chunks
            company_text = json.dumps(company_data, indent=2)
            company_chunks = self.text_splitter.create_documents([company_text])
            
            # Add metadata to chunks
            for chunk in company_chunks:
                chunk.metadata.update({
                    'source': 'company_data',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Create or update company vector store
            if self.vector_stores["company"] is None:
                self.vector_stores["company"] = Chroma.from_documents(
                    documents=company_chunks,
                    embedding=self.embeddings,
                    persist_directory=self.company_store_path
                )
            else:
                # Update existing store with new data
                self.vector_stores["company"].add_documents(company_chunks)
            
            return {
                "status": "success",
                "message": f"Successfully stored {len(company_chunks)} company data chunks",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Error storing company data: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return {"error": error_msg}

    def get_company_data(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant company data based on query"""
        if self.vector_stores["company"] is None:
            return []
            
        try:
            results = self.vector_stores["company"].similarity_search(query, k=k)
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
        except Exception as e:
            print(f"Error retrieving company data: {str(e)}")
            return []

    def load_rfp_document(self, rfp_path: str) -> None:
        print(f"Loading RFP document: {rfp_path}")
        loader = PyPDFLoader(rfp_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['rfp_id'] = "RFP_1"

        rfp_chunks = self.text_splitter.split_documents(docs)
        print(f"Split RFP document into {len(rfp_chunks)} chunks")

        # Create or update RFP vector store
        rfp_store_path = "./chroma_db/rfp"
        self.vector_stores["rfp"] = Chroma.from_documents(
            documents=rfp_chunks,
            embedding=self.embeddings,
            persist_directory=rfp_store_path
        )
        
        bm25_rfp = BM25Retriever.from_documents(rfp_chunks)

        self.ensemble_rfp = EnsembleRetriever(
            retrievers=[self.vector_stores["rfp"].as_retriever(), bm25_rfp],
            weights=[0.7, 0.3]
        )

    def _safe_json_parse(self, raw: str) -> Dict:
        """Enhanced JSON parsing with robust error handling"""
        try:
            cleaned = raw.strip()
            
            # Remove markdown code blocks
            cleaned = re.sub(r'^```json|```$', '', cleaned, flags=re.MULTILINE)
            
            # Ensure proper JSON structure
            if not cleaned.startswith('{'):
                # Handle cases where response starts with array or other structures
                if '"required_sections"' in cleaned:
                    cleaned = '{' + cleaned
                elif cleaned.startswith('['):
                    cleaned = '{"required_sections": ' + cleaned + '}'
                else:
                    cleaned = '{' + cleaned

            if not cleaned.endswith('}'):
                if '"other_details"' in cleaned:
                    cleaned += '}'
                else:
                    cleaned += '}'

            # Fix common JSON errors
            cleaned = re.sub(r',\s*(?=[]}])', '', cleaned)  # Remove trailing commas
            cleaned = re.sub(r"(\w+)\s*:", r'"\1":', cleaned)  # Add quotes around keys
            cleaned = re.sub(r"'(.*?)'", r'"\1"', cleaned)  # Replace single quotes
            cleaned = re.sub(r'\\(")', r'\1', cleaned)  # Remove escaped quotes
            
            # Handle missing commas between objects
            cleaned = re.sub(r'}\s*{', '},{', cleaned)
            
            # Fix malformed time values with colons (e.g., "23":59" -> "23:59")
            cleaned = re.sub(r'"(\d{2})":(\d{2})"', r'"\1:\2"', cleaned)
            
            # Fix double quotes in values
            cleaned = re.sub(r'""([^"]+)"', r'"\1"', cleaned)
            
            # Convert numeric values to strings for specific fields
            cleaned = re.sub(r'"max_pages":\s*(\d+)', r'"max_pages": "\1"', cleaned)
            
            # Fix any remaining double quotes in values
            cleaned = re.sub(r':\s*""([^"]+)"', r': "\1"', cleaned)

            return json.loads(cleaned)
            
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e.msg}")
            print(f"Error at line {e.lineno}, column {e.colno}")
            print(f"Problematic text: {cleaned[e.pos-50:e.pos+50]}")
            return {"error": f"JSON parsing failed: {str(e)}"}
            
        except Exception as e:
            print(f"Unexpected parsing error: {str(e)}")
            print(f"Cleaned JSON: {cleaned}")
            return {"error": f"Unexpected error: {str(e)}"}

    def extract_submission_checklist(self, rfp_id: str) -> Dict:
        print(f"## Extracting Submission Checklist for {rfp_id}")

        try:
            # Retrieve relevant RFP sections
            rfp_docs = self.vector_stores["rfp"].similarity_search(
                "submission requirements formatting deadline sections attachments",
                k=8
            )
            rfp_text = "\n\n".join([doc.page_content for doc in rfp_docs])

            # Get relevant company data for context
            company_docs = self.get_company_data("company capabilities certifications experience")
            company_context = "\n\n".join([doc["content"] for doc in company_docs])

            # Enhanced prompt with example and company context
            checklist_prompt = PromptTemplate.from_template(
                """You are a government contracting expert analyzing an RFP document. Extract ALL submission requirements and format as VALID JSON.
            
            IMPORTANT: Respond ONLY with valid JSON matching this structure:
            {{
                "required_sections": ["section1", ...],
                "formatting_requirements": ["req1", ...],
                "page_limits": {{
                    "max_pages": "number",
                    "specific_section_limits": {{"section": "limit"}}
                }},
                "required_attachments": ["doc1", ...],
                "submission_deadlines": {{
                    "date": "YYYY-MM-DD",
                    "time": "HH:MM",
                    "method": "submission method"
                }},
                "other_details": ["detail1", ...]
            }}
            
            Example of VALID response:
            {{
                "required_sections": ["Technical Proposal", "Budget"],
                "formatting_requirements": ["PDF format", "A4 paper size"],
                "page_limits": {{
                    "max_pages": "50",
                    "specific_section_limits": {{"Technical Proposal": "30"}}
                }},
                "required_attachments": ["Company Registration", "Tax Compliance"],
                "submission_deadlines": {{
                    "date": "2024-06-15",
                    "time": "23:59",
                    "method": "Online portal"
                }},
                "other_details": ["Two hard copies required"]
            }}
            
            RULES:
            1. Use double quotes ONLY
            2. Maintain proper JSON syntax
            3. Start with {{ and end with }}
            4. No markdown or extra text
            5. Use "null" for missing information
            6. Keep arrays empty if no items found
            
            Company Context:
            {company_context}
            
            RFP Text:
            {rfp_text}
            
            Respond ONLY with valid JSON:"""
            )
            
            result = self.llm.invoke(checklist_prompt.format(
                rfp_text=rfp_text,
                company_context=company_context
            ))
            print("Raw LLM response:", result.content)

            structured_checklist = self._safe_json_parse(result.content)
            
            if "error" in structured_checklist:
                raise ValueError(f"JSON parsing failed: {structured_checklist['error']}")

            # Validate required fields
            required_fields = ["required_sections", "formatting_requirements"]
            for field in required_fields:
                if field not in structured_checklist:
                    raise ValueError(f"Missing required field: {field}")

            # Return the structured checklist with metadata
            return {
                "rfp_id": rfp_id,
                "timestamp": datetime.now().isoformat(),
                "checklist": structured_checklist
            }

        except json.JSONDecodeError as e:
            error_msg = f"Critical JSON Error: {e.msg}"
            print(error_msg)
            print(traceback.format_exc())
            return {"error": error_msg}

        except Exception as e:
            error_msg = f"Processing Error: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return {"error": error_msg}

if __name__ == "__main__":
    analyzer = RFPAnalyzer()
    try:
        analyzer.load_rfp_document(r"E:\RFP-Analyzer\ELIGIBLE RFP - 1.pdf")
        result = analyzer.extract_submission_checklist("RFP_1")
        
        if "error" in result:
            print(f"❌ Final Error: {result['error']}")
        else:
            print("✅ Successfully generated submission checklist")
            
    except Exception as e:
        print(f"🔥 Critical Error: {str(e)}")
        print(traceback.format_exc())