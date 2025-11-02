from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
import json
import re
from datetime import datetime
from google import genai
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv  
load_dotenv()


class SearchStrategy(Enum):
    """Enumeration of different search strategies"""
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    CODE = "code"


@dataclass
class RetrievalResult:
    """Data class for retrieval results"""
    content: str
    score: float
    metadata: Dict
    strategy: SearchStrategy


@dataclass
class ValidationResult:
    """Data class for validation results"""
    is_valid: bool
    issues: List[str]
    suggestions: List[str]
    score: float


class PineconeIndex():
    """
    Pinecone Index class for vector store operations.
    """
    def __init__(self, PINECONE_API_KEY, PINECONE_ENVIRONMENT, index_name):
        self.pinecone = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        self.index = self.pinecone.Index(index_name)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def retrieve_index(self, prompt, top_k=5, namespace=None):
        """Retrieve top_k similar items from the index"""
        query_vector = self.embedding_model.encode([prompt]).tolist()[0]
        results = self.index.query(
            vector=query_vector, 
            top_k=top_k, 
            include_metadata=True,
            namespace=namespace
        )
        return results


class MultiStrategyRetrieval():
    """
    Layer 3: Multi-Strategy Retrieval System
    Implements semantic, structural, and code-based search strategies
    """
    def __init__(self, pinecone_index: PineconeIndex, gemini_client: genai.Client, model_name: str):
        self.pinecone_index = pinecone_index
        self.gemini_client = gemini_client
        self.model_name = model_name
        
    def semantic_search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Semantic search using vector embeddings"""
        print("  🔍 Performing semantic search...")
        results = self.pinecone_index.retrieve_index(query, top_k=top_k)
        
        retrieval_results = []
        for match in results.get('matches', []):
            retrieval_results.append(RetrievalResult(
                content=match.get('metadata', {}).get('text', ''),
                score=match.get('score', 0.0),
                metadata=match.get('metadata', {}),
                strategy=SearchStrategy.SEMANTIC
            ))
        return retrieval_results
    
    def structural_search(self, resource_type: str, top_k: int = 3) -> List[RetrievalResult]:
        """Search for structural templates and module patterns"""
        print("  🏗️  Performing structural search...")
        
        query = f"terraform module structure {resource_type} best practices"
        results = self.pinecone_index.retrieve_index(query, top_k=top_k, namespace="templates")
        
        retrieval_results = []
        for match in results.get('matches', []):
            retrieval_results.append(RetrievalResult(
                content=match.get('metadata', {}).get('text', ''),
                score=match.get('score', 0.0),
                metadata=match.get('metadata', {}),
                strategy=SearchStrategy.STRUCTURAL
            ))
        return retrieval_results
    
    def code_search(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """Search for similar code implementations"""
        print("  💻 Performing code search...")
        
        code_query = f"terraform code implementation {query}"
        results = self.pinecone_index.retrieve_index(code_query, top_k=top_k, namespace="code")
        
        retrieval_results = []
        for match in results.get('matches', []):
            retrieval_results.append(RetrievalResult(
                content=match.get('metadata', {}).get('text', ''),
                score=match.get('score', 0.0),
                metadata=match.get('metadata', {}),
                strategy=SearchStrategy.CODE
            ))
        return retrieval_results
    
    def multi_strategy_retrieve(self, query: str, resource_type: str) -> List[RetrievalResult]:
        """Combine all search strategies"""
        print("\n📚 LAYER 3: Multi-Strategy Retrieval\n")
        
        semantic_results = self.semantic_search(query, top_k=5)
        structural_results = self.structural_search(resource_type, top_k=3)
        code_results = self.code_search(query, top_k=3)
        
        all_results = semantic_results + structural_results + code_results
        print(f"  ✓ Retrieved {len(all_results)} total results\n")
        
        return all_results


class IntelligentReranker():
    """
    Layer 4: Intelligent Re-ranking & Validation
    Re-ranks results based on relevance, security, and context
    """
    def __init__(self, gemini_client: genai.Client, model_name: str):
        self.gemini_client = gemini_client
        self.model_name = model_name
    
    def relevance_scoring(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Score results based on relevance to the query"""
        print("  📊 Scoring relevance...")
        
        prompt = f"""Rate the relevance of each document to this query: "{query}"
        
Documents:
{self._format_results_for_scoring(results)}

Respond with JSON array of scores (0-1) for each document:
{{"scores": [0.9, 0.7, ...]}}
"""
        
        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        
        try:
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                scores_data = json.loads(json_match.group())
                scores = scores_data.get('scores', [])
                
                for i, result in enumerate(results):
                    if i < len(scores):
                        result.score = (result.score + scores[i]) / 2
        except:
            pass
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def security_validation(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Validate security aspects of retrieved content"""
        print("  🔒 Validating security...")
        
        validated_results = []
        for result in results:
            content_lower = result.content.lower()
            security_score = 1.0
            
            if 'hardcoded' in content_lower or 'password' in content_lower:
                security_score -= 0.3
            if 'public' in content_lower and 'bucket' in content_lower:
                security_score -= 0.2
            
            if 'encryption' in content_lower or 'kms' in content_lower:
                security_score += 0.1
            if 'iam' in content_lower or 'policy' in content_lower:
                security_score += 0.1
            
            result.score = result.score * security_score
            validated_results.append(result)
        
        return validated_results
    
    def select_best_context(self, results: List[RetrievalResult], max_context: int = 5) -> List[RetrievalResult]:
        """Select the best context from re-ranked results"""
        print("  ✂️  Selecting best context...")
        
        selected = results[:max_context]
        print(f"  ✓ Selected {len(selected)} best results\n")
        
        return selected
    
    def rerank_and_validate(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Complete re-ranking and validation pipeline"""
        print("\n🎯 LAYER 4: Intelligent Re-ranking & Validation\n")
        
        results = self.relevance_scoring(query, results)
        results = self.security_validation(results)
        results = self.select_best_context(results, max_context=5)
        
        return results
    
    def _format_results_for_scoring(self, results: List[RetrievalResult]) -> str:
        """Format results for LLM scoring"""
        formatted = []
        for i, result in enumerate(results):
            formatted.append(f"Document {i+1}:\n{result.content[:300]}...")
        return "\n\n".join(formatted)


class VariableTracker():
    """
    Tracks user variables and ensures they are properly used in generated code
    """
    def __init__(self):
        self.variables = {}
        self.variable_usage_map = {}
    
    def add_variables(self, variables: Dict):
        """Add variables to track"""
        self.variables.update(variables)
        # Initialize usage map
        for key, value in variables.items():
            self.variable_usage_map[key] = {
                'value': value,
                'used': False,
                'locations': []
            }
    
    def check_usage_in_code(self, code: str) -> Tuple[List[str], List[str]]:
        """Check which variables are used/unused in code"""
        used = []
        unused = []
        
        for var_name, var_info in self.variable_usage_map.items():
            var_value = str(var_info['value'])
            
            # Check for exact value match
            if var_value in code:
                used.append(var_name)
                self.variable_usage_map[var_name]['used'] = True
                # Find locations
                lines = code.split('\n')
                for i, line in enumerate(lines, 1):
                    if var_value in line:
                        self.variable_usage_map[var_name]['locations'].append(i)
            else:
                unused.append(var_name)
                self.variable_usage_map[var_name]['used'] = False
        
        return used, unused
    
    def get_usage_report(self) -> str:
        """Generate a detailed usage report"""
        report = []
        for var_name, var_info in self.variable_usage_map.items():
            status = "✓ USED" if var_info['used'] else "✗ MISSING"
            locations = f"(lines: {', '.join(map(str, var_info['locations']))})" if var_info['locations'] else ""
            report.append(f"{status} | {var_name} = '{var_info['value']}' {locations}")
        return '\n'.join(report)
    
    def get_unused_variables(self) -> Dict:
        """Get dictionary of unused variables"""
        return {k: v['value'] for k, v in self.variable_usage_map.items() if not v['used']}


class MultiAgentGeneration():
    """
    Layer 5: Multi-Agent Generation System
    Specialized agents for different aspects of code generation
    """
    def __init__(self, gemini_client: genai.Client, model_name: str):
        self.gemini_client = gemini_client
        self.model_name = model_name
        self.variable_tracker = VariableTracker()
    
    def extract_terraform_code(self, response_text: str) -> str:
        """Extract clean Terraform code from response"""
        code = re.sub(r'```(?:hcl|terraform|tf)?\n?', '', response_text)
        code = re.sub(r'```\n?$', '', code)
        
        lines = code.split('\n')
        clean_lines = []
        in_terraform_block = False
        
        for line in lines:
            stripped = line.strip()
            if any(keyword in stripped for keyword in ['resource', 'variable', 'output', 'data', 'locals', 'terraform', 'provider']):
                in_terraform_block = True
            
            if in_terraform_block:
                if stripped.startswith('#') or stripped.startswith('//'):
                    clean_lines.append(line)
                elif stripped and not any(word in stripped.lower() for word in ['here is', 'this code', 'explanation:', 'note:', 'this will', 'the above', 'this terraform']):
                    clean_lines.append(line)
                elif not stripped:
                    clean_lines.append(line)
        
        return '\n'.join(clean_lines).strip()
    
    def validate_terraform_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Validate basic Terraform syntax"""
        issues = []
        
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            issues.append(f"Unbalanced braces: {open_braces} opening, {close_braces} closing")
        
        if 'resource' in code and not re.search(r'resource\s+"[\w-]+"', code):
            issues.append("Invalid resource syntax")
        
        quotes = code.count('"')
        if quotes % 2 != 0:
            issues.append("Unclosed string quotes detected")
        
        return len(issues) == 0, issues
    
    def create_variable_injection_prompt(self, variables: Dict) -> str:
        """Create detailed instructions for variable injection"""
        instructions = []
        instructions.append("MANDATORY VARIABLE USAGE - EVERY VALUE MUST APPEAR IN CODE:\n")
        
        for var_name, var_value in variables.items():
            # Determine likely Terraform argument name
            tf_arg = var_name.lower().replace(' ', '_').replace('-', '_')
            instructions.append(f"• {var_name} = \"{var_value}\"")
            instructions.append(f"  → MUST use in code as: {tf_arg} = \"{var_value}\"")
            instructions.append(f"  → Required: The exact string \"{var_value}\" MUST appear in the generated code\n")
        
        return '\n'.join(instructions)
    
    def generator_agent(self, query: str, context: List[RetrievalResult], 
                       variables: Dict, max_attempts: int = 3) -> str:
        """Main generator agent with multi-attempt variable enforcement"""
        print("  🤖 Generator Agent: Creating Terraform code...")
        
        # Initialize variable tracker
        self.variable_tracker.add_variables(variables)
        
        context_text = self._format_context(context)
        variable_instructions = self.create_variable_injection_prompt(variables)
        
        for attempt in range(max_attempts):
            if attempt > 0:
                print(f"  🔄 Attempt {attempt + 1}/{max_attempts} - Ensuring all variables are used...")
            
            prompt = f"""You are a Terraform code generator. Generate ONLY valid Terraform HCL code.

User Request: {query}

{variable_instructions}

Reference Documentation (use as guidance for structure):
{context_text}

CRITICAL RULES - FAILURE TO FOLLOW WILL REQUIRE REGENERATION:
1. EVERY user-provided value MUST appear EXACTLY as given in the code
2. DO NOT use placeholder values like "example", "my-bucket", "test" - use the EXACT values provided
3. Return ONLY Terraform code - NO explanations, NO markdown blocks
4. Use proper Terraform HCL syntax with correct indentation (2 spaces)
5. All strings must be properly quoted
6. All brackets must be balanced and closed
7. Include brief inline comments (using #) only for complex logic

VERIFICATION CHECKLIST - Before returning code, verify:
{self._create_verification_checklist(variables)}

Generate the complete Terraform code now:
"""
            
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            terraform_code = self.extract_terraform_code(response.text)
            
            # Check variable usage
            used_vars, unused_vars = self.variable_tracker.check_usage_in_code(terraform_code)
            
            if not unused_vars:
                print(f"  ✓ All {len(used_vars)} variables successfully incorporated")
                return terraform_code
            
            # If variables are missing, regenerate with explicit injection
            if attempt < max_attempts - 1:
                print(f"  ⚠️  Missing {len(unused_vars)} variable(s): {', '.join(unused_vars)}")
                
                # Create targeted fix prompt
                unused_details = '\n'.join([f"  • {var} = \"{variables[var]}\"" for var in unused_vars])
                
                fix_prompt = f"""The following Terraform code is INCOMPLETE. It is MISSING required user values.

INCOMPLETE CODE:
{terraform_code}

MISSING VALUES THAT MUST BE ADDED:
{unused_details}

INSTRUCTIONS:
1. Identify where each missing value should be used in the Terraform resource
2. Add the EXACT values (not placeholders) to the appropriate resource arguments
3. Keep all existing code structure
4. Return ONLY the complete corrected Terraform code
5. NO explanations, NO markdown

EVERY missing value must appear EXACTLY as shown above in the corrected code.

Corrected complete code:
"""
                
                fix_response = self.gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=fix_prompt
                )
                terraform_code = self.extract_terraform_code(fix_response.text)
        
        # Final check after all attempts
        used_vars, unused_vars = self.variable_tracker.check_usage_in_code(terraform_code)
        if unused_vars:
            print(f"  ⚠️  WARNING: {len(unused_vars)} variable(s) still missing after {max_attempts} attempts")
        
        return terraform_code
    
    def _create_verification_checklist(self, variables: Dict) -> str:
        """Create a checklist for the LLM to verify"""
        checklist = []
        for var_name, var_value in variables.items():
            checklist.append(f"  [ ] The exact string \"{var_value}\" appears in the code")
        return '\n'.join(checklist)
    
    def validator_agent(self, terraform_code: str, variables: Dict) -> ValidationResult:
        """Validator agent with comprehensive variable checking"""
        print("  ✅ Validator Agent: Checking correctness...")
        
        # Check variable usage
        used_vars, unused_vars = self.variable_tracker.check_usage_in_code(terraform_code)
        
        issues = []
        if unused_vars:
            for var in unused_vars:
                issues.append(f"CRITICAL: User value '{var} = {variables[var]}' is NOT present in code")
        
        prompt = f"""Review this Terraform code for correctness and completeness:

{terraform_code}

User-Provided Values That MUST Be Present:
{json.dumps(variables, indent=2)}

Already Identified Issues:
{json.dumps(issues, indent=2) if issues else "None"}

Perform additional checks for:
1. Terraform syntax errors
2. Missing required resource arguments
3. Proper resource naming conventions
4. Valid Terraform structure
5. Correct indentation and formatting

Respond in JSON format:
{{
    "is_valid": true/false,
    "issues": ["list of all issues including missing user values"],
    "suggestions": ["improvements"],
    "score": 0.0-1.0,
    "all_user_values_present": true/false
}}
"""
        
        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        
        try:
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                
                # Merge issues
                all_issues = issues + result_data.get('issues', [])
                
                return ValidationResult(
                    is_valid=result_data.get('is_valid', True) and len(unused_vars) == 0,
                    issues=all_issues,
                    suggestions=result_data.get('suggestions', []),
                    score=0.3 if unused_vars else result_data.get('score', 0.8)
                )
        except:
            pass
        
        return ValidationResult(
            is_valid=len(unused_vars) == 0,
            issues=issues,
            suggestions=[],
            score=0.3 if unused_vars else 0.8
        )
    
    def security_agent(self, terraform_code: str) -> ValidationResult:
        """Security agent - identifies security issues"""
        print("  🔐 Security Agent: Analyzing security...")
        
        prompt = f"""Analyze this Terraform code for security issues:

{terraform_code}

Check for:
1. Hardcoded secrets or credentials
2. Public access configurations
3. Missing encryption settings
4. IAM policy issues
5. Network security concerns

Respond in JSON format:
{{
    "is_valid": true/false,
    "issues": ["list of security issues"],
    "suggestions": ["list of security improvements"],
    "score": 0.0-1.0
}}
"""
        
        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        
        try:
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                return ValidationResult(
                    is_valid=result_data.get('is_valid', True),
                    issues=result_data.get('issues', []),
                    suggestions=result_data.get('suggestions', []),
                    score=result_data.get('score', 0.8)
                )
        except:
            pass
        
        return ValidationResult(is_valid=True, issues=[], suggestions=[], score=0.8)
    
    def cost_optimizer_agent(self, terraform_code: str) -> ValidationResult:
        """Cost optimizer agent - suggests cost optimizations"""
        print("  💰 Cost Optimizer Agent: Analyzing costs...")
        
        prompt = f"""Analyze this Terraform code for cost optimization:

{terraform_code}

Check for:
1. Over-provisioned resources
2. Missing cost-saving features
3. Unnecessary data transfer costs
4. Storage optimization opportunities

Respond in JSON format:
{{
    "is_valid": true/false,
    "issues": ["list of cost issues"],
    "suggestions": ["list of cost optimizations"],
    "score": 0.0-1.0
}}
"""
        
        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        
        try:
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                return ValidationResult(
                    is_valid=result_data.get('is_valid', True),
                    issues=result_data.get('issues', []),
                    suggestions=result_data.get('suggestions', []),
                    score=result_data.get('score', 0.8)
                )
        except:
            pass
        
        return ValidationResult(is_valid=True, issues=[], suggestions=[], score=0.8)
    
    def generate_with_agents(self, query: str, context: List[RetrievalResult], 
                            variables: Dict) -> Tuple[str, Dict[str, ValidationResult], VariableTracker]:
        """Orchestrate all agents and return tracker"""
        print("\n🤖 LAYER 5: Multi-Agent Generation\n")
        
        terraform_code = self.generator_agent(query, context, variables, max_attempts=3)
        
        validation_results = {
            'validator': self.validator_agent(terraform_code, variables),
            'security': self.security_agent(terraform_code),
            'cost_optimizer': self.cost_optimizer_agent(terraform_code)
        }
        
        return terraform_code, validation_results, self.variable_tracker
    
    def _format_context(self, context: List[RetrievalResult]) -> str:
        """Format context for generation"""
        formatted = []
        for i, result in enumerate(context):
            formatted.append(f"=== Reference {i+1} (Score: {result.score:.2f}) ===\n{result.content}\n")
        return "\n".join(formatted)


class ReflectionQA():
    """
    Layer 6: Reflection & Quality Assurance with strict variable enforcement
    """
    def __init__(self, gemini_client: genai.Client, model_name: str):
        self.gemini_client = gemini_client
        self.model_name = model_name
    
    def self_critique(self, terraform_code: str, validation_results: Dict[str, ValidationResult], 
                     variables: Dict, variable_tracker: VariableTracker) -> Dict:
        """Perform self-critique with variable usage analysis"""
        print("  🔍 Performing self-critique...")
        
        all_issues = []
        all_suggestions = []
        
        for agent_name, result in validation_results.items():
            all_issues.extend(result.issues)
            all_suggestions.extend(result.suggestions)
        
        # Get unused variables from tracker
        unused_vars = variable_tracker.get_unused_variables()
        
        prompt = f"""Critically review this Terraform code for quality and completeness:

{terraform_code}

User-Provided Values (ALL MUST be present in code):
{json.dumps(variables, indent=2)}

Variable Usage Status:
{variable_tracker.get_usage_report()}

Known Issues:
{json.dumps(all_issues, indent=2)}

Suggestions:
{json.dumps(all_suggestions[:5], indent=2)}

Provide comprehensive critique in JSON format:
{{
    "overall_quality": 0.0-1.0,
    "strengths": ["list of strengths"],
    "weaknesses": ["list of weaknesses"],
    "must_fix": ["CRITICAL issues that must be fixed"],
    "improvements": ["suggested improvements"],
    "all_variables_used": true/false,
    "missing_variables": ["list of variable names not in code"]
}}
"""
        
        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        
        try:
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                critique = json.loads(json_match.group())
                
                # Add missing variables to must_fix
                if unused_vars:
                    critique['all_variables_used'] = False
                    critique['missing_variables'] = list(unused_vars.keys())
                    for var_name, var_value in unused_vars.items():
                        critique['must_fix'] = critique.get('must_fix', []) + [
                            f"CRITICAL: Add user value {var_name} = '{var_value}' to the code"
                        ]
                else:
                    critique['all_variables_used'] = True
                    critique['missing_variables'] = []
                
                return critique
        except:
            pass
        
        has_unused = len(unused_vars) > 0
        return {
            "overall_quality": 0.4 if has_unused else 0.8,
            "strengths": [],
            "weaknesses": [f"Missing variable: {k}" for k in unused_vars.keys()],
            "must_fix": [f"Add {k} = '{v}'" for k, v in unused_vars.items()],
            "improvements": [],
            "all_variables_used": not has_unused,
            "missing_variables": list(unused_vars.keys())
        }
    
    def iterative_refinement(self, terraform_code: str, critique: Dict, 
                            context: List[RetrievalResult], variables: Dict, 
                            variable_tracker: VariableTracker) -> str:
        """Refine code with laser focus on missing variables"""
        print("  ✨ Performing iterative refinement...")
        
        if critique.get('all_variables_used', False) and not critique.get('must_fix'):
            return terraform_code
        
        unused_vars = variable_tracker.get_unused_variables()
        
        if not unused_vars:
            return terraform_code
        
        context_text = self._format_context(context)
        
        # Create explicit instructions for each missing variable
        missing_var_instructions = []
        for var_name, var_value in unused_vars.items():
            missing_var_instructions.append(
                f"• {var_name}: \"{var_value}\" - Find the appropriate Terraform argument and set it to this EXACT value"
            )
        
        prompt = f"""CRITICAL FIX REQUIRED: The following Terraform code is MISSING required user values.

INCOMPLETE CODE:
{terraform_code}

MISSING VALUES (MUST ADD ALL OF THESE):
{chr(10).join(missing_var_instructions)}

ALL USER VALUES FOR REFERENCE:
{json.dumps(variables, indent=2)}

REFERENCE DOCUMENTATION:
{context_text}

INSTRUCTIONS:
1. Identify which Terraform resource arguments need these values
2. Add the EXACT values (not placeholders) to the code
3. Maintain all existing code structure and other values
4. Return ONLY the corrected Terraform code
5. NO explanations, NO markdown, NO comments about changes

VERIFICATION: Before returning, ensure EVERY value from "MISSING VALUES" appears in the code.

Corrected code with ALL user values:
"""
        
        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        
        refined_code = re.sub(r'```(?:hcl|terraform|tf)?\n?', '', response.text)
        refined_code = re.sub(r'```\n?$', '', refined_code).strip()
        
        return refined_code
    
    def reflection_qa_pipeline(self, terraform_code: str, validation_results: Dict[str, ValidationResult],
                               context: List[RetrievalResult], variables: Dict, 
                               variable_tracker: VariableTracker,
                               max_iterations: int = 4) -> str:
        """Complete reflection and QA pipeline with aggressive variable enforcement"""
        print("\n🔄 LAYER 6: Reflection & Quality Assurance\n")
        
        current_code = terraform_code
        
        for iteration in range(max_iterations):
            print(f"  🔁 Iteration {iteration + 1}/{max_iterations}")
            
            # Update tracker with current code
            variable_tracker.variable_usage_map = {}
            variable_tracker.add_variables(variables)
            used_vars, unused_vars = variable_tracker.check_usage_in_code(current_code)
            
            # Self-critique
            critique = self.self_critique(current_code, validation_results, variables, variable_tracker)
            
            print(f"    Variables used: {len(used_vars)}/{len(variables)}")
            if unused_vars:
                print(f"    Missing: {', '.join(unused_vars)}")
            
            # Check if all variables are used and quality is good
            if (critique.get('all_variables_used', False) and 
                critique.get('overall_quality', 0) >= 0.85 and
                not critique.get('must_fix')):
                print(f"  ✓ All variables incorporated (quality: {critique.get('overall_quality')})")
                break
            
            # If variables are missing or quality is low, refine
            if unused_vars or critique.get('must_fix'):
                current_code = self.iterative_refinement(
                    current_code, critique, context, variables, variable_tracker
                )
        
        # Final verification
        variable_tracker.variable_usage_map = {}
        variable_tracker.add_variables(variables)
        used_vars, unused_vars = variable_tracker.check_usage_in_code(current_code)
        
        if unused_vars:
            print(f"  ⚠️  WARNING: {len(unused_vars)} variable(s) still missing after {max_iterations} iterations")
        else:
            print(f"  ✓ SUCCESS: All {len(variables)} variables properly incorporated")
        
        print(f"  ✓ Reflection complete\n")
        
        return current_code
    
    def _format_context(self, context: List[RetrievalResult]) -> str:
        """Format context"""
        formatted = []
        for i, result in enumerate(context):
            formatted.append(f"Reference {i+1}:\n{result.content[:500]}...")
        return "\n\n".join(formatted)


class RAGSystem():
    """
    Complete RAG System with all layers integrated and strict variable enforcement
    """
    def __init__(self, pinecone_index: PineconeIndex, gemini_client: genai.Client, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.pinecone_index = pinecone_index 
        self.gemini_client = gemini_client
        
        self.retrieval = MultiStrategyRetrieval(pinecone_index, gemini_client, model_name)
        self.reranker = IntelligentReranker(gemini_client, model_name)
        self.agents = MultiAgentGeneration(gemini_client, model_name)
        self.reflection = ReflectionQA(gemini_client, model_name)

    def query_understanding_agent(self, user_query: str) -> Dict:
        """Layer 1: Query Understanding with value extraction"""
        print("\n🧠 LAYER 1: Query Understanding\n")
        print("  Analyzing your request...\n")
        
        prompt = f"""Analyze this Terraform infrastructure request and extract ALL specific details:

User Request: {user_query}

Extract:
1. Resource type (e.g., S3 bucket, EC2 instance, VPC, RDS)
2. EVERY specific value mentioned (names, IDs, regions, sizes, counts, etc.)
3. Additional variables that should be asked from the user

Be thorough - extract EVERY concrete value the user provided.

Respond in JSON format:
{{
    "resource_type": "primary AWS resource type",
    "user_provided_values": {{
        "descriptive_key": "exact_value_from_query",
        "another_key": "another_value"
    }},
    "required_variables": ["list", "of", "additional", "needed", "info"],
    "optional_configs": ["list", "of", "optional", "settings"]
}}

Example:
If user says: "Create an S3 bucket named my-app-bucket in us-west-2 with versioning"
Return:
{{
    "resource_type": "S3 bucket",
    "user_provided_values": {{
        "bucket_name": "my-app-bucket",
        "region": "us-west-2",
        "versioning": "enabled"
    }},
    "required_variables": [],
    "optional_configs": ["encryption", "lifecycle_rules"]
}}
"""
        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        
        try:
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                requirements = json.loads(json_match.group())
                print(f"  ✓ Resource type: {requirements.get('resource_type')}")
                print(f"  ✓ Values extracted: {len(requirements.get('user_provided_values', {}))}")
                
                # Display extracted values
                if requirements.get('user_provided_values'):
                    print("\n  📋 Extracted from your query:")
                    for key, value in requirements.get('user_provided_values', {}).items():
                        print(f"    • {key}: {value}")
                
                print(f"  ✓ Additional info needed: {len(requirements.get('required_variables', []))}\n")
                return requirements
        except Exception as e:
            print(f"  ⚠️  Error parsing response: {e}")
        
        return {
            "resource_type": "infrastructure",
            "user_provided_values": {},
            "required_variables": [],
            "optional_configs": []
        }
    
    def collect_user_variables(self, requirements: Dict) -> Dict:
        """Layer 2: Variable Collection with validation"""
        print("🔗 LAYER 2: Collecting Required Information\n")
        
        # Start with values already extracted
        user_variables = requirements.get('user_provided_values', {}).copy()
        
        if not user_variables and not requirements.get('required_variables'):
            print("  ⚠️  No specific values provided. Please provide details for better results.\n")
        
        # Ask for additional required variables
        required_vars = requirements.get('required_variables', [])
        if required_vars:
            print("Please provide the following required information:\n")
            for var in required_vars:
                while True:
                    value = input(f"  {var}: ").strip()
                    if value:
                        user_variables[var] = value
                        break
                    else:
                        print(f"    ⚠️  This field is required. Please provide a value.")
        
        # Ask for optional configurations
        optional_configs = requirements.get('optional_configs', [])
        if optional_configs:
            print("\n📝 Optional configurations (press Enter to skip):\n")
            for config in optional_configs:
                value = input(f"  {config} [optional]: ").strip()
                if value:
                    user_variables[config] = value
        
        # Validate that we have at least some variables
        if not user_variables:
            print("\n  ⚠️  WARNING: No specific values provided!")
            print("  The generated code will be generic and may not meet your needs.")
            print("  Consider providing specific names, regions, or configurations.\n")
        else:
            print(f"\n  ✓ Total variables collected: {len(user_variables)}")
            print("\n  📦 Your configuration:")
            for key, value in user_variables.items():
                print(f"    • {key}: {value}")
            print()
        
        return user_variables
    
    def generate_terraform_code(self, user_query: str) -> Dict:
        """Complete pipeline from query to final code with strict variable enforcement"""
        print("\n" + "="*70)
        print("🚀 TERRAFORM IaC GENERATION PIPELINE")
        print("="*70)
        
        # Layer 1: Query Understanding
        requirements = self.query_understanding_agent(user_query)
        
        # Layer 2: Variable Collection
        variables = self.collect_user_variables(requirements)
        
        if not variables:
            print("\n⚠️  Proceeding with generic code generation...")
            print("Consider re-running with specific values for better results.\n")
        
        # Layer 3: Multi-Strategy Retrieval
        retrieval_results = self.retrieval.multi_strategy_retrieve(
            user_query, 
            requirements['resource_type']
        )
        
        # Layer 4: Re-ranking & Validation
        best_context = self.reranker.rerank_and_validate(user_query, retrieval_results)
        
        # Layer 5: Multi-Agent Generation with variable tracking
        terraform_code, validation_results, variable_tracker = self.agents.generate_with_agents(
            user_query, 
            best_context, 
            variables
        )
        
        # Layer 6: Reflection & QA with aggressive variable enforcement
        final_code = self.reflection.reflection_qa_pipeline(
            terraform_code,
            validation_results,
            best_context,
            variables,
            variable_tracker,
            max_iterations=4
        )
        
        # Final comprehensive check
        final_tracker = VariableTracker()
        final_tracker.add_variables(variables)
        used_vars, unused_vars = final_tracker.check_usage_in_code(final_code)
        
        # Print results
        self._print_results(final_code, validation_results, best_context, variables, final_tracker)
        
        return {
            'terraform_code': final_code,
            'validation_results': validation_results,
            'requirements': requirements,
            'variables': variables,
            'retrieved_context': best_context,
            'variable_tracker': final_tracker,
            'used_variables': used_vars,
            'unused_variables': unused_vars
        }
    
    def _print_results(self, terraform_code: str, 
                       validation_results: Dict[str, ValidationResult],
                       context: List[RetrievalResult],
                       variables: Dict,
                       variable_tracker: VariableTracker):
        """Print final results with detailed variable analysis"""
        print("\n" + "="*70)
        print("📄 GENERATED TERRAFORM CODE")
        print("="*70)
        print(terraform_code)
        
        print("\n" + "="*70)
        print("🔍 VARIABLE USAGE VERIFICATION")
        print("="*70)
        
        if variables:
            print(f"\nTotal variables provided: {len(variables)}")
            print("\nDetailed usage analysis:\n")
            print(variable_tracker.get_usage_report())
            
            unused = variable_tracker.get_unused_variables()
            if unused:
                print(f"\n⚠️  CRITICAL WARNING: {len(unused)} variable(s) NOT used in code!")
                print("\nMissing variables:")
                for var_name, var_value in unused.items():
                    print(f"  ✗ {var_name} = '{var_value}'")
                print("\n⚠️  Please manually add these values to the generated code.")
            else:
                print(f"\n✅ SUCCESS: All {len(variables)} variables properly incorporated!")
        else:
            print("\n⚠️  No variables were provided - code is generic")
        
        print("\n" + "="*70)
        print("📊 VALIDATION SUMMARY")
        print("="*70)
        
        for agent_name, result in validation_results.items():
            print(f"\n{agent_name.upper()}:")
            print(f"  Valid: {'✓ Yes' if result.is_valid else '✗ No'}")
            print(f"  Score: {result.score:.2f}")
            if result.issues:
                print(f"  Issues ({len(result.issues)}):")
                for issue in result.issues[:3]:
                    print(f"    • {issue}")
                if len(result.issues) > 3:
                    print(f"    ... and {len(result.issues) - 3} more")
            if result.suggestions:
                print(f"  Suggestions ({len(result.suggestions)}):")
                for suggestion in result.suggestions[:2]:
                    print(f"    • {suggestion}")
        
        print("\n" + "="*70)
        print("📚 RETRIEVED CONTEXT USED")
        print("="*70)
        print(f"  ✓ Used {len(context)} reference documents")
        for i, ctx in enumerate(context[:3]):
            print(f"  {i+1}. {ctx.strategy.value.upper()} (Score: {ctx.score:.2f})")
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE")
        print("="*70)


def main():
    """Main execution function"""
    print("🔧 Initializing Terraform IaC RAG System...\n")
    
    # Load environment variables
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    
    if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
        print("❌ Error: PINECONE_API_KEY and PINECONE_ENVIRONMENT must be set")
        return
    
    # Initialize Gemini client
    try:
        gemini_client = genai.Client()
    except Exception as e:
        print(f"❌ Error initializing Gemini client: {e}")
        print("Make sure GEMINI_API_KEY environment variable is set")
        return
    
    # Initialize vector store
    try:
        vector_store = PineconeIndex(
            PINECONE_API_KEY, 
            PINECONE_ENVIRONMENT, 
            index_name="iac-terraform-v2"
        )
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        return
    
    # Initialize RAG system
    rag_system = RAGSystem(
        pinecone_index=vector_store,
        gemini_client=gemini_client,
        model_name="gemini-2.5-flash"
    )
    
    # Get user query
    print("="*70)
    print("🎯 TERRAFORM INFRASTRUCTURE CODE GENERATOR")
    print("="*70)
    print("\n💡 Tips for best results:")
    print("  • Be specific with names (e.g., 'my-prod-bucket' not 'a bucket')")
    print("  • Include regions (e.g., 'us-east-1', 'eu-west-2')")
    print("  • Mention sizes/types (e.g., 't2.micro', '100GB')")
    print("  • Specify configurations (e.g., 'with encryption', 'public access')")
    print("\n📝 Examples:")
    print("  • 'Create an S3 bucket named data-lake-prod in us-west-2 with versioning'")
    print("  • 'Deploy an EC2 instance named web-server type t2.micro in us-east-1'")
    print("  • 'Create RDS MySQL database named myapp-db in eu-central-1'")
    print()
    
    user_query = input("📝 Describe the infrastructure you want to create:\n> ")
    
    if not user_query.strip():
        print("❌ Error: Query cannot be empty")
        return
    
    # Generate Terraform code
    result = rag_system.generate_terraform_code(user_query)
    
    # Automatically save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"terraform_code_{timestamp}.tf"
    
    with open(filename, "w") as f:
        f.write(result['terraform_code'])
    
    print(f"\n💾 Terraform code saved to: {filename}")
    
    # Save metadata
    metadata_filename = f"terraform_metadata_{timestamp}.json"
    metadata = {
        'query': user_query,
        'requirements': result['requirements'],
        'variables': result['variables'],
        'used_variables': result['used_variables'],
        'unused_variables': result['unused_variables'],
        'variable_usage_complete': len(result['unused_variables']) == 0,
        'validation_summary': {
            agent: {
                'is_valid': val.is_valid,
                'score': val.score,
                'issues_count': len(val.issues),
                'suggestions_count': len(val.suggestions)
            }
            for agent, val in result['validation_results'].items()
        },
        'retrieved_documents_count': len(result['retrieved_context']),
        'timestamp': timestamp
    }
    
    with open(metadata_filename, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Metadata saved to: {metadata_filename}")
    
    # Final recommendations
    print("\n" + "="*70)
    if result['unused_variables']:
        print("⚠️  ACTION REQUIRED:")
        print(f"  {len(result['unused_variables'])} variable(s) are missing from the code")
        print(f"  Please review {filename} and manually add:")
        for var in result['unused_variables']:
            print(f"    • {var} = '{result['variables'][var]}'")
        print(f"\n  These values must be added for the code to work correctly!")
    else:
        if result['variables']:
            print("✅ ALL VARIABLES SUCCESSFULLY INCORPORATED!")
            print(f"  All {len(result['variables'])} user-provided values are in the code")
        else:
            print("⚠️  GENERIC CODE GENERATED")
            print("  No specific values were provided")
            print("  Consider re-running with specific names and configurations")
    
    print("="*70)
    print(f"\n🎉 Generation complete! Check {filename} for your Terraform code.")


if __name__ == "__main__":
    main()