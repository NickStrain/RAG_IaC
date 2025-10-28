"""
Integrated RAG-based Terraform Generation System with Validation Feedback Loop
Complete system with generator and validator in one file.
"""

import os
from typing import List, Dict, Optional
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import pathlib
import shutil
import subprocess
import json
import re

load_dotenv()

# ---------------------------
# Environment Variables
# ---------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = "terraform-iac-v1"  

# Initialize APIs
genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ============================================================================
# VALIDATOR AGENT
# ============================================================================

class TerraformValidator:
    """Comprehensive Terraform code validator"""
    
    def __init__(self, terraform_file: str):
        self.terraform_file = pathlib.Path(terraform_file)
        self.terraform_dir = self.terraform_file.parent
        self.validation_results = {
            "syntax_check": {},
            "security_check": {},
            "llm_review": {},
            "overall_status": "PENDING"
        }
    
    def validate_all(self) -> Dict:
        """Run all validation checks"""
        print("\n" + "="*60)
        print("🔍 TERRAFORM VALIDATION AGENT")
        print("="*60)
        
        # 1. Syntax Validation
        print("\n[1/4] Running syntax validation...")
        self.validation_results["syntax_check"] = self._validate_syntax()
        
        # 2. Security Best Practices
        print("\n[2/4] Checking security best practices...")
        self.validation_results["security_check"] = self._check_security()
        
        # 3. LLM-based Code Review
        print("\n[3/4] Running LLM code review...")
        self.validation_results["llm_review"] = self._llm_code_review()
        
        # 4. Generate Overall Status
        print("\n[4/4] Generating overall assessment...")
        self._generate_overall_status()
        
        # Print Summary
        self._print_summary()
        
        return self.validation_results
    
    def _validate_syntax(self) -> Dict:
        """Validate Terraform syntax using terraform validate"""
        result = {
            "status": "UNKNOWN",
            "message": "",
            "details": []
        }
        
        if not self._is_terraform_installed():
            result["status"] = "SKIPPED"
            result["message"] = "Terraform CLI not found. Install Terraform to run syntax validation."
            return result
        
        try:
            # Initialize terraform
            init_result = subprocess.run(
                ["terraform", "init", "-backend=false"],
                cwd=self.terraform_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Run terraform validate
            validate_result = subprocess.run(
                ["terraform", "validate", "-json"],
                cwd=self.terraform_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            try:
                validation_output = json.loads(validate_result.stdout)
                if validation_output.get("valid", False):
                    result["status"] = "PASS"
                    result["message"] = "✅ Terraform syntax is valid"
                else:
                    result["status"] = "FAIL"
                    result["message"] = "❌ Terraform validation failed"
                    result["details"] = validation_output.get("diagnostics", [])
            except json.JSONDecodeError:
                result["status"] = "ERROR"
                result["message"] = f"Failed to parse terraform output: {validate_result.stdout}"
                
        except subprocess.TimeoutExpired:
            result["status"] = "ERROR"
            result["message"] = "Terraform validation timed out"
        except Exception as e:
            result["status"] = "ERROR"
            result["message"] = f"Error during validation: {str(e)}"
        
        return result
    
    def _check_security(self) -> Dict:
        """Check for common security issues"""
        result = {
            "status": "PASS",
            "issues_found": [],
            "warnings": []
        }
        
        try:
            terraform_code = self.terraform_file.read_text(encoding="utf-8")
        except Exception as e:
            result["status"] = "ERROR"
            result["issues_found"].append(f"Could not read file: {str(e)}")
            return result
        
        security_patterns = [
            {
                "pattern": r'encryption\s*=\s*false',
                "issue": "Encryption disabled",
                "severity": "HIGH"
            },
            {
                "pattern": r'public_access_block\s*=\s*false',
                "issue": "Public access block disabled",
                "severity": "HIGH"
            },
            {
                "pattern": r'acl\s*=\s*["\']public-read',
                "issue": "Public read ACL configured",
                "severity": "HIGH"
            },
            {
                "pattern": r'versioning\s*{[^}]*enabled\s*=\s*false',
                "issue": "Versioning disabled",
                "severity": "MEDIUM"
            },
            {
                "pattern": r'(aws_access_key|aws_secret|password)\s*=\s*["\'][^"\']+["\']',
                "issue": "Hardcoded credentials detected",
                "severity": "CRITICAL"
            },
        ]
        
        for check in security_patterns:
            matches = re.finditer(check["pattern"], terraform_code, re.IGNORECASE | re.DOTALL)
            for match in matches:
                issue = {
                    "severity": check["severity"],
                    "issue": check["issue"],
                    "location": f"Line {terraform_code[:match.start()].count(chr(10)) + 1}"
                }
                result["issues_found"].append(issue)
                if check["severity"] in ["HIGH", "CRITICAL"]:
                    result["status"] = "FAIL"
        
        if result["status"] == "PASS" and not result["issues_found"]:
            result["message"] = "✅ No security issues detected"
        else:
            result["message"] = f"❌ Found {len(result['issues_found'])} security issues"
        
        return result
    
    def _llm_code_review(self) -> Dict:
        """Use Gemini for code review"""
        result = {
            "status": "PENDING",
            "review": "",
            "suggestions": [],
            "issues": [],
            "security_concerns": [],
            "score": 0
        }
        
        try:
            terraform_code = self.terraform_file.read_text(encoding="utf-8")
            
            prompt = f"""
You are an expert Terraform and AWS infrastructure reviewer. Analyze the following Terraform code.

TERRAFORM CODE:
```hcl
{terraform_code}
```

Provide your review in JSON format:
{{
    "overall_score": <1-10>,
    "issues": ["list of problems"],
    "suggestions": ["list of improvements"],
    "security_concerns": ["list of security issues"],
    "summary": "brief assessment"
}}
"""
            
            llm = genai.GenerativeModel("gemini-2.5-flash")
            response = llm.generate_content(prompt)
            
            review_text = response.text if hasattr(response, "text") else str(response)
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', review_text, re.DOTALL)
            if json_match:
                review_json = json.loads(json_match.group())
                result["score"] = review_json.get("overall_score", 5)
                result["suggestions"] = review_json.get("suggestions", [])
                result["issues"] = review_json.get("issues", [])
                result["security_concerns"] = review_json.get("security_concerns", [])
                result["review"] = review_json.get("summary", review_text)
            else:
                result["review"] = review_text
                result["score"] = 5
            
            # Determine status
            if result["score"] >= 8:
                result["status"] = "EXCELLENT"
            elif result["score"] >= 6:
                result["status"] = "GOOD"
            elif result["score"] >= 4:
                result["status"] = "NEEDS_IMPROVEMENT"
            else:
                result["status"] = "POOR"
                
        except Exception as e:
            result["status"] = "ERROR"
            result["review"] = f"Error during LLM review: {str(e)}"
        
        return result
    
    def _generate_overall_status(self):
        """Generate overall validation status"""
        syntax = self.validation_results["syntax_check"].get("status")
        security = self.validation_results["security_check"].get("status")
        llm = self.validation_results["llm_review"].get("status")
        
        if syntax == "FAIL" or security == "FAIL":
            self.validation_results["overall_status"] = "FAILED"
        elif llm in ["POOR", "NEEDS_IMPROVEMENT"]:
            self.validation_results["overall_status"] = "NEEDS_IMPROVEMENT"
        elif syntax == "PASS" and security == "PASS":
            self.validation_results["overall_status"] = "PASSED"
        else:
            self.validation_results["overall_status"] = "PARTIAL"
    
    def _print_summary(self):
        """Print validation summary"""
        print("\n" + "="*60)
        print("📊 VALIDATION SUMMARY")
        print("="*60)
        
        status = self.validation_results["overall_status"]
        status_emoji = "✅" if status == "PASSED" else "⚠️" if status == "PARTIAL" else "❌"
        print(f"\n{status_emoji} Overall Status: {status}")
        
        print(f"\n🔧 Syntax: {self.validation_results['syntax_check'].get('status')}")
        print(f"🔒 Security: {self.validation_results['security_check'].get('status')}")
        print(f"🤖 LLM Review: {self.validation_results['llm_review'].get('status')} (Score: {self.validation_results['llm_review'].get('score', 0)}/10)")
        
        issues = self.validation_results['security_check'].get('issues_found', [])
        if issues:
            print(f"\n⚠️  Security Issues Found: {len(issues)}")
            for issue in issues[:3]:
                print(f"   • [{issue['severity']}] {issue['issue']}")
        
        print("="*60)
    
    def _is_terraform_installed(self) -> bool:
        """Check if terraform CLI is available"""
        try:
            subprocess.run(["terraform", "version"], capture_output=True, timeout=5)
            return True
        except:
            return False


# ============================================================================
# GENERATOR AGENT
# ============================================================================

def get_embedding(text: str) -> List[float]:
    """Generate embedding vector"""
    return embedder.encode(text).tolist()


def retrieve_docs(query: str, top_k: int = 5):
    """Retrieve relevant Terraform docs from Pinecone"""
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results


def generate_terraform_code(query: str, retrieved_docs, validation_feedback: Optional[Dict] = None, previous_code: Optional[str] = None) -> str:
    """Generate Terraform code with optional validation feedback"""
    matches = retrieved_docs.matches if hasattr(retrieved_docs, 'matches') else []
    
    context = "\n".join(
        [match.metadata.get("text", "") if hasattr(match, 'metadata') else "" for match in matches]
    )

    if validation_feedback and previous_code:
        feedback_summary = _format_validation_feedback(validation_feedback)
        
        prompt = f"""
You are an expert in Terraform and AWS Infrastructure as Code.

PREVIOUS ATTEMPT HAD ISSUES. You must fix the following problems:

{feedback_summary}

PREVIOUS CODE (WITH ERRORS):
```hcl
{previous_code}
```

Using the following Terraform documentation:
{context}

Original user request: '{query}'

Generate a CORRECTED, production-ready Terraform configuration that:
1. FIXES all validation errors mentioned above
2. ADDRESSES all security issues
3. IMPLEMENTS all suggestions from the code review
4. Follows AWS and Terraform best practices
5. Includes proper comments

IMPORTANT: Only provide the corrected Terraform code without any additional explanations.
"""
    else:
        prompt = f"""
You are an expert in Terraform and AWS Infrastructure as Code.

Using the following retrieved Terraform documentation snippets:
{context}

Generate a complete, production-ready Terraform configuration that satisfies this user request:
'{query}'

Make sure:
- The Terraform code follows best practices
- Uses appropriate AWS resources and variables
- Includes comments for clarity
- Implements security best practices (encryption, private access, versioning)
- Uses proper resource naming and tagging
- Just provide the Terraform code without any additional explanations
"""

    llm = genai.GenerativeModel("gemini-2.5-flash")
    response = llm.generate_content(prompt)
    
    terraform_text = ""
    if hasattr(response, "text") and response.text:
        terraform_text = response.text
    elif getattr(response, "output", None):
        terraform_text = getattr(response, "output_text", "") or str(response.output)
    else:
        terraform_text = str(response)

    terraform_text = _clean_code_output(terraform_text)
    return terraform_text


def _format_validation_feedback(validation_results: Dict) -> str:
    """Format validation results into readable feedback"""
    feedback_parts = []
    
    # Syntax errors
    syntax = validation_results.get("syntax_check", {})
    if syntax.get("status") == "FAIL":
        feedback_parts.append("SYNTAX ERRORS:")
        for detail in syntax.get("details", [])[:5]:
            feedback_parts.append(f"  - {detail.get('summary', 'Unknown error')}")
    
    # Security issues
    security = validation_results.get("security_check", {})
    issues = security.get("issues_found", [])
    if issues:
        feedback_parts.append("\nSECURITY ISSUES:")
        for issue in issues[:10]:
            feedback_parts.append(f"  - [{issue['severity']}] {issue['issue']} at {issue['location']}")
    
    # LLM review
    llm_review = validation_results.get("llm_review", {})
    if llm_review.get("issues"):
        feedback_parts.append("\nCODE REVIEW ISSUES:")
        for issue in llm_review["issues"][:5]:
            feedback_parts.append(f"  - {issue}")
    
    if llm_review.get("suggestions"):
        feedback_parts.append("\nIMPROVEMENT SUGGESTIONS:")
        for suggestion in llm_review["suggestions"][:5]:
            feedback_parts.append(f"  - {suggestion}")
    
    if llm_review.get("security_concerns"):
        feedback_parts.append("\nSECURITY CONCERNS:")
        for concern in llm_review["security_concerns"][:5]:
            feedback_parts.append(f"  - {concern}")
    
    return "\n".join(feedback_parts) if feedback_parts else "Minor improvements needed."


def _clean_code_output(code: str) -> str:
    """Remove markdown code blocks"""
    code = code.replace("```hcl", "").replace("```terraform", "").replace("```", "")
    return code.strip()


def save_terraform_to_file(terraform_code: str, base_dir: str = "generated", filename: str = "main.tf") -> str:
    """Save terraform code to file"""
    out_dir = pathlib.Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(terraform_code, encoding="utf-8")
    print(f"✅ Written Terraform to: {out_path.resolve()}")

    if shutil.which("terraform"):
        try:
            subprocess.run(["terraform", "fmt", str(out_path)], check=True, capture_output=True)
            print("✅ Ran `terraform fmt` on the generated file.")
        except subprocess.CalledProcessError as e:
            print("⚠️ terraform fmt failed:", e.stderr.decode() if e.stderr else e)

    return str(out_path.resolve())


# ============================================================================
# MAIN FEEDBACK LOOP
# ============================================================================

def rag_generate_terraform_with_validation(query: str, out_dir: str = "generated", max_iterations: int = 3):
    """Generate Terraform code with validation feedback loop"""
    print("\n" + "="*70)
    print("🚀 TERRAFORM GENERATION WITH VALIDATION FEEDBACK LOOP")
    print("="*70)
    
    iteration = 0
    validation_results = None
    terraform_code = None
    saved_path = None
    retrieved_docs = None
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"🔄 ITERATION {iteration}/{max_iterations}")
        print(f"{'='*70}")
        
        # Step 1: Generate or Regenerate Code
        if iteration == 1:
            print(f"\n🔍 Query: {query}")
            retrieved_docs = retrieve_docs(query)
            matches_count = len(retrieved_docs.matches) if hasattr(retrieved_docs, 'matches') else 0
            print(f"✅ Retrieved {matches_count} relevant documents.")
            
            terraform_code = generate_terraform_code(query, retrieved_docs)
            print("\n💡 Generated initial Terraform code")
        else:
            print(f"\n🔧 Regenerating code based on validation feedback...")
            terraform_code = generate_terraform_code(
                query, 
                retrieved_docs, 
                validation_feedback=validation_results,
                previous_code=terraform_code
            )
            print("✅ Regenerated Terraform code with fixes")
        
        # Step 2: Save the code
        saved_path = save_terraform_to_file(terraform_code, base_dir=out_dir, filename="main.tf")
        
        # Step 3: Validate the code
        print("\n🔍 Running validation checks...")
        validator = TerraformValidator(saved_path)
        validation_results = validator.validate_all()
        
        # Step 4: Check if validation passed
        overall_status = validation_results.get("overall_status", "UNKNOWN")
        
        if overall_status == "PASSED":
            print("\n" + "="*70)
            print("✅ SUCCESS! Terraform code passed all validations!")
            print("="*70)
            break
        elif iteration < max_iterations:
            print(f"\n⚠️ Validation found issues. Attempting to fix... ({iteration}/{max_iterations})")
        else:
            print("\n" + "="*70)
            print(f"❌ Maximum iterations ({max_iterations}) reached.")
            print("Code generated but may still have issues. Manual review required.")
            print("="*70)
    
    # Final Summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    print(f"Total iterations: {iteration}")
    print(f"Final status: {validation_results.get('overall_status', 'UNKNOWN')}")
    print(f"Output file: {saved_path}")
    
    # Save iteration history
    history_file = pathlib.Path(out_dir) / "generation_history.json"
    history = {
        "query": query,
        "iterations": iteration,
        "final_status": validation_results.get("overall_status"),
        "validation_results": validation_results
    }
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f"Generation history saved to: {history_file}")
    
    return {
        "terraform_file": saved_path,
        "validation_results": validation_results,
        "iterations": iteration,
        "success": overall_status == "PASSED"
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    user_query = "Create an S3 bucket with versioning and server-side encryption"
    
    result = rag_generate_terraform_with_validation(
        user_query, 
        out_dir="generated",
        max_iterations=3
    )
    
    if result["success"]:
        print("\n🎉 Terraform code is ready for deployment!")
    else:
        print("\n⚠️ Please review the generated code manually before deployment.")