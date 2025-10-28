import ollama 
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
import json
import re
from dotenv import load_dotenv  
load_dotenv()

class RAGSystemWithChaining:
    def __init__(self, pinecone_api_key, index_name="terraform-iac-v1", 
                 generation_model="codellama:7b-instruct"):
        """
        Initialize RAG system with Pinecone and Ollama with prompt chaining
        
        Args:
            pinecone_api_key: Your Pinecone API key
            index_name: Name of your Pinecone index
            generation_model: Ollama model to use for text generation
        """
        self.generation_model = generation_model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        pc = Pinecone(api_key=pinecone_api_key)
        self.index = pc.Index(index_name)
        
        print(f"✓ Connected to Pinecone index: {index_name}")
        print(f"✓ Using embedding model: all-MiniLM-L6-v2")
        print(f"✓ Using generation model: {generation_model}\n")
    
    def get_embedding(self, text):
        """Generate embeddings using SentenceTransformer"""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def retrieve_documents(self, query, top_k=3):
        """Retrieve relevant documents from Pinecone"""
        query_embedding = self.get_embedding(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results['matches']
    
    def format_context(self, documents):
        """Format retrieved documents into context string"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.get('metadata', {})
            text = metadata.get('text', 'No content available')
            score = doc.get('score', 0)
            context_parts.append(f"[Document {i}] (Relevance: {score:.3f})\n{text}")
        return "\n\n".join(context_parts)
    
    def extract_requirements(self, user_query):
        """
        Step 1: Extract what needs to be created and identify missing information
        """
        print("🔗 STEP 1: Analyzing your request...\n")
        
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
        
        response = ollama.generate(model=self.generation_model, prompt=prompt)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response['response'], re.DOTALL)
            if json_match:
                requirements = json.loads(json_match.group())
            else:
                # Fallback parsing
                requirements = {
                    "resource_type": "infrastructure",
                    "required_variables": [],
                    "optional_configs": [],
                    "clarification_needed": True
                }
        except:
            requirements = {
                "resource_type": "infrastructure",
                "required_variables": [],
                "optional_configs": [],
                "clarification_needed": True
            }
        
        print(f"📋 Resource Type: {requirements['resource_type']}")
        print(f"📋 Required Variables: {', '.join(requirements['required_variables']) if requirements['required_variables'] else 'None identified'}")
        print(f"📋 Optional Configs: {', '.join(requirements['optional_configs']) if requirements['optional_configs'] else 'None'}\n")
        
        return requirements
    
    def collect_user_variables(self, requirements):
        """
        Step 2: Interactively collect required variables from user
        """
        print("🔗 STEP 2: Collecting required information...\n")
        
        user_variables = {}
        
        if not requirements['required_variables']:
            print("✓ No additional variables needed\n")
            return user_variables
        
        print("Please provide the following information:\n")
        
        for var in requirements['required_variables']:
            value = input(f"  {var}: ").strip()
            user_variables[var] = value
        
        # Ask about optional configurations
        if requirements['optional_configs']:
            print("\n📝 Optional configurations (press Enter to skip):\n")
            for config in requirements['optional_configs']:
                value = input(f"  {config} [optional]: ").strip()
                if value:
                    user_variables[config] = value
        
        print(f"\n✓ Collected {len(user_variables)} variable(s)\n")
        return user_variables
    
    def enrich_query_with_context(self, original_query, requirements, user_variables):
        """
        Step 3: Create enriched query with all collected information
        """
        print("🔗 STEP 3: Enriching query with collected information...\n")
        
        enriched_parts = [
            f"Resource Type: {requirements['resource_type']}",
            f"Original Request: {original_query}"
        ]
        
        if user_variables:
            enriched_parts.append("\nUser-Provided Variables:")
            for key, value in user_variables.items():
                enriched_parts.append(f"  - {key}: {value}")
        
        enriched_query = "\n".join(enriched_parts)
        
        print("✓ Enriched query created\n")
        return enriched_query
    
    def generate_terraform_code(self, enriched_query, context, user_variables):
        """
        Step 4: Generate final Terraform code with all context
        """
        print("🔗 STEP 4: Generating Terraform code...\n")
        
        variables_section = ""
        if user_variables:
            variables_section = "\n\nUser Variables:\n"
            for key, value in user_variables.items():
                variables_section += f"  {key} = {value}\n"
        
        augmented_prompt = f"""You are a DevOps assistant expert in Terraform IaC.
Generate ONLY Terraform code. Do not include explanations.

Retrieved Context from Knowledge Base:
{context}

{enriched_query}
{variables_section}

Generate the complete Terraform code:
"""
        
        response = ollama.generate(
            model=self.generation_model,
            prompt=augmented_prompt
        )
        
        return response['response']
    
    def query_with_chaining(self, user_query, top_k=3, auto_collect=True):
        """
        Complete RAG pipeline with prompt chaining
        
        Args:
            user_query: User's initial question
            top_k: Number of documents to retrieve
            auto_collect: If True, interactively collect variables; if False, return requirements
            
        Returns:
            Dictionary with answer, variables, and documents
        """
        print(f"\n{'='*70}")
        print(f"🚀 Starting Prompt Chain RAG Pipeline")
        print(f"{'='*70}\n")
        print(f"Initial Query: {user_query}\n")
        
        # CHAIN STEP 1: Extract requirements
        requirements = self.extract_requirements(user_query)
        
        # CHAIN STEP 2: Collect variables
        if auto_collect:
            user_variables = self.collect_user_variables(requirements)
        else:
            return {
                'requirements': requirements,
                'status': 'awaiting_user_input'
            }
        
        # CHAIN STEP 3: Enrich query
        enriched_query = self.enrich_query_with_context(
            user_query, requirements, user_variables
        )
        
        # CHAIN STEP 4: Retrieve relevant documents
        print("🔗 STEP 5: Retrieving relevant documentation...\n")
        documents = self.retrieve_documents(enriched_query, top_k=top_k)
        context = self.format_context(documents)
        print(f"✓ Retrieved {len(documents)} relevant documents\n")
        
        # CHAIN STEP 5: Generate final code
        terraform_code = self.generate_terraform_code(
            enriched_query, context, user_variables
        )
        
        print(f"{'='*70}")
        print("✅ GENERATED TERRAFORM CODE")
        print(f"{'='*70}\n")
        print(terraform_code)
        print(f"\n{'='*70}\n")
        
        return {
            'terraform_code': terraform_code,
            'requirements': requirements,
            'user_variables': user_variables,
            'documents': documents,
            'status': 'completed'
        }
    
    def batch_query_with_variables(self, user_query, variables_dict, top_k=3):
        """
        Non-interactive mode: provide variables directly
        
        Args:
            user_query: User's question
            variables_dict: Dictionary of pre-defined variables
            top_k: Number of documents to retrieve
        """
        print(f"\n{'='*70}")
        print(f"🚀 Batch Mode RAG Pipeline")
        print(f"{'='*70}\n")
        
        requirements = self.extract_requirements(user_query)
        enriched_query = self.enrich_query_with_context(
            user_query, requirements, variables_dict
        )
        
        documents = self.retrieve_documents(enriched_query, top_k=top_k)
        context = self.format_context(documents)
        
        terraform_code = self.generate_terraform_code(
            enriched_query, context, variables_dict
        )
        
        print(f"{'='*70}")
        print("✅ GENERATED TERRAFORM CODE")
        print(f"{'='*70}\n")
        print(terraform_code)
        print(f"\n{'='*70}\n")
        
        return {
            'terraform_code': terraform_code,
            'requirements': requirements,
            'user_variables': variables_dict,
            'documents': documents
        }


# Example usage
if __name__ == "__main__":
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-api-key-here")
    
    rag = RAGSystemWithChaining(
        pinecone_api_key=PINECONE_API_KEY,
        index_name="terraform-iac-v1"
    )
    
    # Example 1: Interactive mode with prompt chaining
    print("="*70)
    print("EXAMPLE 1: Interactive Mode")
    print("="*70)
    
    result = rag.query_with_chaining(
        "Create an S3 bucket with versioning and encryption"
    )
    
    # Example 2: Batch mode with pre-defined variables
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Mode")
   