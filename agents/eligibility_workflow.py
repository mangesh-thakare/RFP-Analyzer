from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Dict, Optional
import tempfile
import os
import sys
import shutil
import PyPDF2

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.eligibility_check_agent import analyze_rfp_endpoint
from agents.eligibility_matcher_agent import EligibilityMatcher
from utils.text_extraction import extract_text_from_pdf

app = FastAPI()

class EligibilityWorkflow:
    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.matcher = EligibilityMatcher(groq_api_key=groq_api_key)

    def categorize_mandatory_requirements(self, requirements: Dict) -> Dict:
        """Categorize mandatory requirements into subsections"""
        categories = {
            "Registrations_and_Licenses": {
                "Requirements": {},
                "Remarks": []
            },
            "Certifications": {
                "Requirements": {},
                "Remarks": []
            },
            "Policies_and_Compliance": {
                "Requirements": {},
                "Remarks": []
            }
        }
        
        # Keywords for categorization
        registration_keywords = ["registration", "license", "sam.gov", "w-9", "employment agency"]
        certification_keywords = ["certification", "certified", "certificate"]
        
        for req, met in requirements.items():
            req_lower = req.lower()
            
            # Determine category
            if any(keyword in req_lower for keyword in registration_keywords):
                category = "Registrations_and_Licenses"
            elif any(keyword in req_lower for keyword in certification_keywords):
                category = "Certifications"
            else:
                category = "Policies_and_Compliance"
            
            # Add requirement to appropriate category
            categories[category]["Requirements"][req] = {
                "Status": met,
                "Remarks": "Missing requirement" if not met else "None"
            }
            
            # Add to category remarks if not met
            if not met:
                categories[category]["Remarks"].append(f"Missing: {req}")
        
        # Set "None" for remarks if all requirements are met
        for category in categories.values():
            if not category["Remarks"]:
                category["Remarks"] = ["None"]
            else:
                category["Remarks"] = sorted(category["Remarks"])
        
        return categories

    async def process_rfp(self, file: UploadFile) -> Dict:
        """Process RFP document and check eligibility"""
        temp_path = None
        try:
            # Save the uploaded file
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

            file.file.seek(0)
            check_result = await analyze_rfp_endpoint(file)
            
            if check_result["status"] != "success":
                raise HTTPException(
                    status_code=400, 
                    detail=f"Failed to analyze RFP: {check_result.get('error', 'Unknown error')}"
                )

            requirements = check_result["requirements"]
            analysis = self.matcher.check_eligibility(requirements)

            # Process insurance requirements using original requirements for coverage details
            insurance_reqs = {}
            seen_insurance = set()
            original_insurance = requirements.get("insurance_requirements", {})
            for ins_type, required_coverage in original_insurance.items():
                normalized_type = ins_type.lower().replace('_', ' ')
                if normalized_type in seen_insurance:
                    continue
                seen_insurance.add(normalized_type)
                details = analysis["insurance_requirements"]["details"].get(ins_type, {})
                insurance_reqs[ins_type] = {
                    "Status": {
                        "met": details.get("met", False),
                        "required": required_coverage
                    },
                    "Remarks": details.get("remark", "Not analyzed")
                }

            # Process optional requirements
            optional_reqs = {}
            seen_optional = set()
            for req, met in analysis["optional_requirements"]["met"].items():
                normalized_req = req.lower().replace('may submit', '').replace('may provide', '').strip()
                if normalized_req not in seen_optional and not req.startswith('eg.'):
                    optional_reqs[req] = {
                        "Status": met,
                        "Remarks": "None"
                    }
                    seen_optional.add(normalized_req)

            # Categorize mandatory requirements
            mandatory_categories = self.categorize_mandatory_requirements(
                analysis["mandatory_requirements"]["met"]
            )

            response = {
                "Eligibility_Analysis": {
                    "1. Mandatory_Requirements": {
                        "A. Registrations_and_Licenses": {
                            "Requirements": mandatory_categories["Registrations_and_Licenses"]["Requirements"],
                            "Section_Remarks": mandatory_categories["Registrations_and_Licenses"]["Remarks"]
                        },
                        "B. Certifications": {
                            "Requirements": mandatory_categories["Certifications"]["Requirements"],
                            "Section_Remarks": mandatory_categories["Certifications"]["Remarks"]
                        },
                        "C. Policies_and_Compliance": {
                            "Requirements": mandatory_categories["Policies_and_Compliance"]["Requirements"],
                            "Section_Remarks": mandatory_categories["Policies_and_Compliance"]["Remarks"]
                        }
                    },
                    "2. Experience_Requirements": {
                        "Requirements": {
                            "Met": analysis["experience_requirements"]["met"],
                            "Details": analysis["experience_requirements"]["details"]
                        },
                        "Section_Remarks": ["Experience requirement not met"] if not analysis["experience_requirements"]["met"] else ["None"]
                    },
                    "3. Insurance_Requirements": {
                        "Requirements": insurance_reqs,
                        "Section_Remarks": ["Verify insurance limits"] if any("Verify specific limits" in req["Remarks"] 
                                         for req in insurance_reqs.values()) else ["None"]
                    },
                    "4. Optional_Requirements": {
                        "Requirements": optional_reqs,
                        "Section_Remarks": ["None"]
                    }
                }
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

@app.post("/analyze-rfp-eligibility/")
async def analyze_rfp_eligibility(file: UploadFile = File(...)):
    """Analyze RFP and check company eligibility"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    return await workflow.process_rfp(file)

@app.get("/")
async def root():
    return {"message": "RFP Eligibility Analysis API"}

# Initialize workflow
workflow = EligibilityWorkflow()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)