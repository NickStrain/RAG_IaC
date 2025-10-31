import ollama 
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
    def __init__(self, pinecone_index: PineconeIndex, ollama_model: str):
        self.pinecone_index = pinecone_index
        self.ollama_model = ollama_model
        
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
        
        # Search for resource-specific templates
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
        
        # Search specifically in code namespace
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
    def __init__(self, ollama_model: str):
        self.ollama_model = ollama_model
    
    def relevance_scoring(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Score results based on relevance to the query"""
        print("  📊 Scoring relevance...")
        
        prompt = f"""Rate the relevance of each document to this query: "{query}"
        
Documents:
{self._format_results_for_scoring(results)}

Respond with JSON array of scores (0-1) for each document:
{{"scores": [0.9, 0.7, ...]}}
"""
        
        response = ollama.generate(model=self.ollama_model, prompt=prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response['response'], re.DOTALL)
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
            # Check for security best practices
            content_lower = result.content.lower()
            security_score = 1.0
            
            # Penalize insecure patterns
            if 'hardcoded' in content_lower or 'password' in content_lower:
                security_score -= 0.3
            if 'public' in content_lower and 'bucket' in content_lower:
                security_score -= 0.2
            
            # Reward secure patterns
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
        
        # Return top N results
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


class MultiAgentGeneration():
    """
    Layer 5: Multi-Agent Generation System
    Specialized agents for different aspects of code generation
    """
    def __init__(self, ollama_model: str):
        self.ollama_model = ollama_model
    
    def generator_agent(self, query: str, context: List[RetrievalResult], 
                       variables: Dict) -> str:
        """Main generator agent - creates Terraform code"""
        print("  🤖 Generator Agent: Creating Terraform code...")
        
        context_text = self._format_context(context)
        variables_text = json.dumps(variables, indent=2)
        
        prompt = f"""You are a Terraform expert. Generate clean, production-ready Terraform code.

User Request: {query}

User Variables:
{variables_text}

Reference Context:
{context_text}

Generate complete Terraform code with:
1. Proper resource definitions
2. Variable declarations
3. Output definitions
4. Comments explaining key decisions

Return ONLY the Terraform code, no explanations.
"""
        
        response = ollama.generate(model=self.ollama_model, prompt=prompt)
        return response['response']
    
    def validator_agent(self, terraform_code: str) -> ValidationResult:
        """Validator agent - checks code correctness"""
        print("  ✅ Validator Agent: Checking correctness...")
        
        prompt = f"""Review this Terraform code for correctness:

{terraform_code}

Check for:
1. Syntax errors
2. Missing required arguments
3. Proper resource naming
4. Variable usage

Respond in JSON format:
{{
    "is_valid": true/false,
    "issues": ["list of issues"],
    "suggestions": ["list of improvements"],
    "score": 0.0-1.0
}}
"""
        
        response = ollama.generate(model=self.ollama_model, prompt=prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response['response'], re.DOTALL)
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
    
    def security_agent(self, terraform_code: str) -> ValidationResult:
        """Security agent - identifies security issues"""
        print("  🔐 Security Agent: Analyzing security...")
        
        prompt = f"""Analyze this Terraform code for security issues:

{terraform_code}

Check for:
1. Hardcoded secrets
2. Public access configurations
3. Missing encryption
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
        
        response = ollama.generate(model=self.ollama_model, prompt=prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response['response'], re.DOTALL)
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
2. Missing cost-saving features (spot instances, reserved capacity)
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
        
        response = ollama.generate(model=self.ollama_model, prompt=prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response['response'], re.DOTALL)
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
                            variables: Dict) -> Tuple[str, Dict[str, ValidationResult]]:
        """Orchestrate all agents"""
        print("\n🤖 LAYER 5: Multi-Agent Generation\n")
        
        # Generate code
        terraform_code = self.generator_agent(query, context, variables)
        
        # Validate with specialized agents
        validation_results = {
            'validator': self.validator_agent(terraform_code),
            'security': self.security_agent(terraform_code),
            'cost_optimizer': self.cost_optimizer_agent(terraform_code)
        }
        
        return terraform_code, validation_results
    
    def _format_context(self, context: List[RetrievalResult]) -> str:
        """Format context for generation"""
        formatted = []
        for i, result in enumerate(context):
            formatted.append(f"Reference {i+1}:\n{result.content}")
        return "\n\n".join(formatted)


class ReflectionQA():
    """
    Layer 6: Reflection & Quality Assurance
    Self-critique and iterative refinement
    """
    def __init__(self, ollama_model: str):
        self.ollama_model = ollama_model
    
    def self_critique(self, terraform_code: str, validation_results: Dict[str, ValidationResult]) -> Dict:
        """Perform self-critique on generated code"""
        print("  🔍 Performing self-critique...")
        
        all_issues = []
        all_suggestions = []
        
        for agent_name, result in validation_results.items():
            all_issues.extend(result.issues)
            all_suggestions.extend(result.suggestions)
        
        prompt = f"""Perform a critical review of this Terraform code:

{terraform_code}

Known Issues:
{json.dumps(all_issues, indent=2)}

Suggestions:
{json.dumps(all_suggestions, indent=2)}

Provide a comprehensive critique in JSON format:
{{
    "overall_quality": 0.0-1.0,
    "strengths": ["list of strengths"],
    "weaknesses": ["list of weaknesses"],
    "must_fix": ["critical issues that must be fixed"],
    "improvements": ["suggested improvements"]
}}
"""
        
        response = ollama.generate(model=self.ollama_model, prompt=prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response['response'], re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "overall_quality": 0.8,
            "strengths": [],
            "weaknesses": [],
            "must_fix": [],
            "improvements": []
        }
    
    def iterative_refinement(self, terraform_code: str, critique: Dict, 
                            context: List[RetrievalResult], variables: Dict) -> str:
        """Refine code based on critique"""
        print("  ✨ Performing iterative refinement...")
        
        if not critique.get('must_fix') and not critique.get('improvements'):
            return terraform_code
        
        context_text = self._format_context(context)
        variables_text = json.dumps(variables, indent=2)
        
        prompt = f"""Refine this Terraform code based on the critique:

Original Code:
{terraform_code}

Critique:
{json.dumps(critique, indent=2)}

Reference Context:
{context_text}

User Variables:
{variables_text}

Generate improved Terraform code that addresses all issues and improvements.
Return ONLY the refined Terraform code.
"""
        
        response = ollama.generate(model=self.ollama_model, prompt=prompt)
        return response['response']
    
    def generate_tests(self, terraform_code: str) -> str:
        """Generate test cases for the Terraform code"""
        print("  🧪 Generating test cases...")
        
        prompt = f"""Generate Terratest or basic test cases for this Terraform code:

{terraform_code}

Create tests that verify:
1. Resources are created correctly
2. Outputs are accessible
3. Security configurations are correct
4. Resource dependencies work

Return test code in Go (Terratest) or a test plan.
"""
        
        response = ollama.generate(model=self.ollama_model, prompt=prompt)
        return response['response']
    
    def reflection_qa_pipeline(self, terraform_code: str, validation_results: Dict[str, ValidationResult],
                               context: List[RetrievalResult], variables: Dict, 
                               max_iterations: int = 2) -> Tuple[str, str]:
        """Complete reflection and QA pipeline"""
        print("\n🔄 LAYER 6: Reflection & Quality Assurance\n")
        
        current_code = terraform_code
        
        for iteration in range(max_iterations):
            print(f"  🔁 Iteration {iteration + 1}/{max_iterations}")
            
            # Self-critique
            critique = self.self_critique(current_code, validation_results)
            
            # Check if refinement is needed
            if critique.get('overall_quality', 0) >= 0.9 and not critique.get('must_fix'):
                print(f"  ✓ Code quality sufficient (score: {critique.get('overall_quality')})")
                break
            
            # Refine
            current_code = self.iterative_refinement(current_code, critique, context, variables)
        
        # Generate tests
        test_code = self.generate_tests(current_code)
        
        print(f"  ✓ Reflection complete\n")
        
        return current_code, test_code
    
    def _format_context(self, context: List[RetrievalResult]) -> str:
        """Format context"""
        formatted = []
        for i, result in enumerate(context):
            formatted.append(f"Reference {i+1}:\n{result.content[:500]}...")
        return "\n\n".join(formatted)


class RAGSystem():
    """
    Complete RAG System with all layers integrated
    """
    def __init__(self, pinecone_index: PineconeIndex, genai_api_key: str, ollama_model_name: str):
        self.ollama_model_name = ollama_model_name
        self.pinecone_index = pinecone_index 
        self.genai_client = genai.Client(api_key=genai_api_key)
        
        # Initialize all layers
        self.retrieval = MultiStrategyRetrieval(pinecone_index, ollama_model_name)
        self.reranker = IntelligentReranker(ollama_model_name)
        self.agents = MultiAgentGeneration(ollama_model_name)
        self.reflection = ReflectionQA(ollama_model_name)

    def query_understanding_agent(self, user_query: str) -> Dict:
        """Layer 1: Query Understanding"""
        print("\n🧠 LAYER 1: Query Understanding\n")
        print("  Analyzing your request...\n")
        
        prompt = f"""Analyze this Terraform infrastructure request and extract:
1. Resource type (e.g., S3 bucket, EC2 instance, VPC)
2. Required variables that MUST be collected from the user
3. Optional configurations mentioned

User Request: {user_query}

Respond in this exact JSON format:
{{
    "resource_type": "the main resource to create",
    "required_variables": ["list", "of", "required", "variables"],
    "optional_configs": ["list", "of", "optional", "settings"],
    "clarification_needed": true/false
}}
"""
        response = ollama.generate(model="codellama:7b-instruct", prompt=prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response['response'], re.DOTALL)
            if json_match:
                requirements = json.loads(json_match.group())
                print(f"  ✓ Resource type: {requirements.get('resource_type')}")
                print(f"  ✓ Required variables: {len(requirements.get('required_variables', []))}\n")
                return requirements
        except:
            pass
        
        return {
            "resource_type": "infrastructure",
            "required_variables": [],
            "optional_configs": [],
            "clarification_needed": True
        }
    
    def collect_user_variables(self, requirements: Dict) -> Dict:
        """Layer 2: Variable Collection"""
        print("🔗 LAYER 2: Collecting Required Information\n")
        
        user_variables = {}
        
        if not requirements['required_variables']:
            print("  ✓ No additional variables needed\n")
            return user_variables
        
        print("Please provide the following information:\n")
        
        for var in requirements['required_variables']:
            value = input(f"  {var}: ").strip()
            user_variables[var] = value
        
        if requirements['optional_configs']:
            print("\n📝 Optional configurations (press Enter to skip):\n")
            for config in requirements['optional_configs']:
                value = input(f"  {config} [optional]: ").strip()
                if value:
                    user_variables[config] = value
        
        print(f"\n  ✓ Collected {len(user_variables)} variable(s)\n")
        return user_variables
    
    def generate_terraform_code(self, user_query: str) -> Dict:
        """Complete pipeline from query to final code"""
        print("\n" + "="*70)
        print("🚀 TERRAFORM IaC GENERATION PIPELINE")
        print("="*70)
        
        # Layer 1: Query Understanding
        requirements = self.query_understanding_agent(user_query)
        
        # Layer 2: Variable Collection
        variables = self.collect_user_variables(requirements)
        
        # Layer 3: Multi-Strategy Retrieval
        retrieval_results = self.retrieval.multi_strategy_retrieve(
            user_query, 
            requirements['resource_type']
        )
        
        # Layer 4: Re-ranking & Validation
        best_context = self.reranker.rerank_and_validate(user_query, retrieval_results)
        
        # Layer 5: Multi-Agent Generation
        terraform_code, validation_results = self.agents.generate_with_agents(
            user_query, 
            best_context, 
            variables
        )
        
        # Layer 6: Reflection & QA
        final_code, test_code = self.reflection.reflection_qa_pipeline(
            terraform_code,
            validation_results,
            best_context,
            variables
        )
        
        # Print results
        self._print_results(final_code, test_code, validation_results)
        
        return {
            'terraform_code': final_code,
            'test_code': test_code,
            'validation_results': validation_results,
            'requirements': requirements,
            'variables': variables
        }
    
    def _print_results(self, terraform_code: str, test_code: str, 
                       validation_results: Dict[str, ValidationResult]):
        """Print final results"""
        print("\n" + "="*70)
        print("📄 GENERATED TERRAFORM CODE")
        print("="*70)
        print(terraform_code)
        
        print("\n" + "="*70)
        print("📊 VALIDATION SUMMARY")
        print("="*70)
        
        for agent_name, result in validation_results.items():
            print(f"\n{agent_name.upper()}:")
            print(f"  Valid: {result.is_valid}")
            print(f"  Score: {result.score:.2f}")
            if result.issues:
                print(f"  Issues: {len(result.issues)}")
                for issue in result.issues[:3]:
                    print(f"    - {issue}")
            if result.suggestions:
                print(f"  Suggestions: {len(result.suggestions)}")
                for suggestion in result.suggestions[:3]:
                    print(f"    - {suggestion}")
        
        print("\n" + "="*70)
        print("🧪 GENERATED TESTS")
        print("="*70)
        print(test_code[:500] + "..." if len(test_code) > 500 else test_code)
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE")
        print("="*70)


def main():
    """Main execution function"""
    print("🔧 Initializing Terraform IaC RAG System...\n")
    
    # Load environment variables
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    GENAI_API_KEY = os.getenv("GENAI_API_KEY")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "codellama:7b-instruct")
    
    # Initialize vector store
    vector_store = PineconeIndex(
        PINECONE_API_KEY, 
        PINECONE_ENVIRONMENT, 
        index_name="iac-terraform-v2"
    )
    
    # Initialize RAG system
    rag_system = RAGSystem(
        pinecone_index=vector_store,
        genai_api_key=GENAI_API_KEY,
        ollama_model_name=OLLAMA_MODEL
    )
    
    # Example query
    user_query = "create an S3 bucket with versioning enabled and lifecycle policies for cost optimization"
    
    # Generate Terraform code
    result = rag_system.generate_terraform_code(user_query)
    
    # Optionally save to files
    save_option = input("\n💾 Save generated code to files? (y/n): ").strip().lower()
    if save_option == 'y':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(f"terraform_code_{timestamp}.tf", "w") as f:
            f.write(result['terraform_code'])
        
        with open(f"test_code_{timestamp}.go", "w") as f:
            f.write(result['test_code'])
        
        print(f"\n✅ Files saved with timestamp: {timestamp}")


if __name__ == "__main__":
    main()