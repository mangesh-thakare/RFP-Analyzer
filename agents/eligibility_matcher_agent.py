from typing import Dict, List, Optional
import json
from groq import Groq
from agents.document_store import DocumentStore

class EligibilityMatcher:
    def __init__(self, groq_api_key: str):
        self.doc_store = DocumentStore()
        self.groq_client = Groq(api_key=groq_api_key)
        
    def _query_llm(self, prompt: str) -> str:
        """Helper method to query Groq LLM"""
        try:
            completion = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error querying LLM: {str(e)}")
            return "Unable to verify - Error in processing"

    def _analyze_requirement(self, requirement: str, context: str) -> Dict:
        """Analyze a single requirement with LLM"""
        prompt = f"""Analyze this requirement against the provided context. 
        Return ONLY a JSON object with: 
        - 'met' (boolean)
        - 'remark' (short explanation from context)
        
        Requirement: {requirement}
        Context: {context}"""
        
        try:
            llm_response = self._query_llm(prompt)
            return json.loads(llm_response)
        except:
            return {"met": False, "remark": "Requirement analysis failed"}

    def _check_mandatory_requirements(self, requirements: Dict) -> Dict:
        """Check mandatory requirements with LLM analysis"""
        result = {"met": {}, "details": {}}
        
        for req_type in ["registrations_and_licenses", "certifications"]:
            for req in requirements.get(req_type, []):
                if req.lower() in ["none mentioned", "none specified"]:
                    continue
                
                # Get relevant context from documents
                context = "\n".join(
                    [doc["content"] for doc in 
                     self.doc_store.search_text(req, num_results=3)]
                )
                
                # Analyze with LLM
                analysis = self._analyze_requirement(req, context)
                
                result["met"][req] = analysis.get("met", False)
                result["details"][req] = analysis.get("remark", "No analysis available")

        return result

    def _check_experience_requirements(self, required_years: int) -> Dict:
        """Check experience requirements with LLM analysis"""
        context = "\n".join(
            [doc["content"] for doc in 
             self.doc_store.search_text("experience years", num_results=5)]
        )
        
        prompt = f"""Analyze if the company meets {required_years} years experience requirement.
        Context: {context}
        Return JSON with: 'met' (boolean), 'remark' (explanation), 'evidence_years' (number or null)"""
        
        try:
            llm_response = self._query_llm(prompt)
            analysis = json.loads(llm_response)
            return {
                "met": analysis.get("met", False),
                "details": analysis.get("remark", "Experience analysis failed"),
                "company_years": analysis.get("evidence_years", 0)
            }
        except:
            return {
                "met": False,
                "details": "Experience verification failed",
                "company_years": 0
            }

    def _check_insurance_requirements(self, requirements: Dict) -> Dict:
        """Check insurance requirements with LLM analysis"""
        result = {"met": True, "details": {}}
        context = "\n".join(
            [doc["content"] for doc in 
             self.doc_store.search_text("insurance coverage", num_results=5)]
        )
        
        for ins_type, req_details in requirements.items():
            prompt = f"""Verify {ins_type} insurance coverage for {req_details['coverage']}.
            Context: {context}
            Return JSON with: 'met' (boolean), 'remark' (explanation)"""
            
            try:
                llm_response = self._query_llm(prompt)
                analysis = json.loads(llm_response)
                result["details"][ins_type] = {
                    "met": analysis.get("met", False),
                    "remark": analysis.get("remark", "Insurance analysis failed")
                }
                if not analysis.get("met", False):
                    result["met"] = False
            except:
                result["details"][ins_type] = {
                    "met": False,
                    "remark": "Insurance verification failed"
                }
                result["met"] = False

        return result
    def check_eligibility(self, requirements: Dict) -> Dict:
        """Main method to check all eligibility requirements"""
        result = {
            "mandatory_requirements": self._check_mandatory_requirements(requirements),
            "experience_requirements": self._check_experience_requirements(
                requirements.get("experience_years", 0)
            ),
            "insurance_requirements": self._check_insurance_requirements(
                requirements.get("insurance_requirements", {})
            ),
            "optional_requirements": {
                "met": {},
                "details": {}
            }
        }
        
        # Check optional requirements if any exist
        if "optional_requirements" in requirements and requirements["optional_requirements"]:
            for req in requirements["optional_requirements"]:
                # Get relevant context from documents
                context = "\n".join(
                    [doc["content"] for doc in 
                     self.doc_store.search_text(req, num_results=3)]
                )
                
                # Analyze with LLM
                analysis = self._analyze_requirement(req, context)
                
                result["optional_requirements"]["met"][req] = analysis.get("met", False)
                result["optional_requirements"]["details"][req] = analysis.get("remark", "No analysis available")
        
        return result

    # Similar LLM-powered updates for other check methods
    # ... (rest of the class methods updated similarly)

# Example usage with Groq integration
if __name__ == "__main__":
    try:
        # Initialize with Groq API key
        matcher = EligibilityMatcher(groq_api_key="your_groq_api_key_here")
        
        # Load requirements and perform analysis
        with open("requirements.json", "r") as f:
            requirements = json.load(f)["requirements"]
        
        analysis = matcher.check_eligibility(requirements)
        
        # Print dynamic remarks from analysis
        print(json.dumps(analysis, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")