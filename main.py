"""
Pinecone RAG System with HuggingFace - Terraform Code Generator
Retrieval Augmented Generation system optimized for generating Terraform code
Index: terraform-iac-v1
"""

import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class TerraformCodeGenerator:
    def __init__(self, pinecone_api_key, embedding_model="sentence-transformers/all-MiniLM-L6-v2", 
                 llm_model="Salesforce/codegen-350M-mono"):
        """
        Initialize the Terraform Code Generator with Pinecone and HuggingFace models
        
        Args:
            pinecone_api_key: Your Pinecone API key
            embedding_model: HuggingFace embedding model name
            llm_model: HuggingFace code generation model name
        """
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = "terraform-iac-v1"
        
        # Connect to existing index
        self.index = self.pc.Index(self.index_name)
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Load LLM for code generation
        print(f"Loading Code Generation LLM: {llm_model}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        print(f"Models loaded successfully on {self.device}")
    
    def get_index_stats(self):
        """Get statistics about the Pinecone index"""
        return self.index.describe_index_stats()
    
    def create_embedding(self, text):
        """
        Create embedding vector from text using HuggingFace model
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def search_similar_code(self, query, top_k=5, namespace=""):
        """
        Search for similar Terraform code examples in Pinecone
        
        Args:
            query: Search query text
            top_k: Number of results to return
            namespace: Pinecone namespace (optional)
            
        Returns:
            List of matches with metadata
        """
        # Convert query to embedding
        query_embedding = self.create_embedding(query)
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        
        return results
    
    def extract_terraform_code(self, metadata):
        """
        Extract Terraform code from metadata
        
        Args:
            metadata: Document metadata from Pinecone
            
        Returns:
            Extracted Terraform code string
        """
        # Try different possible keys for code content
        code_keys = ['code', 'terraform_code', 'content', 'text', 'source_code']
        
        for key in code_keys:
            if key in metadata and metadata[key]:
                return metadata[key]
        
        # If no code found, return the full metadata as string
        return str(metadata)
    
    def generate_terraform_code(self, query, context_docs):
        """
        Generate Terraform code using retrieved examples as context
        
        Args:
            query: User's request for Terraform code
            context_docs: Retrieved document matches from vector search
            
        Returns:
            Generated Terraform code string
        """
        # Extract Terraform code examples from retrieved documents
        code_examples = []
        for i, match in enumerate(context_docs[:3]):  # Top 3 most relevant
            code = self.extract_terraform_code(match.metadata)
            if code and len(code) > 10:  # Only include substantial code
                code_examples.append(f"# Example {i+1}:\n{code}")
        
        # Build context with code examples
        context = "\n\n".join(code_examples) if code_examples else "# No examples found"
        
        # Create prompt optimized for code generation
        prompt = f"""# Terraform Code Generator
# Task: {query}

# Reference Examples from documentation:
{context}

# Generated Terraform Configuration:
"""
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.3,  # Lower temperature for more deterministic code
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode generated code
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        generated_code = generated_text[len(prompt):].strip()
        
        return generated_code
    
    def generate(self, request, top_k=5, namespace=""):
        """
        Complete RAG pipeline: search for relevant code + generate new code
        
        Args:
            request: User's request for Terraform code
            top_k: Number of documents to retrieve
            namespace: Pinecone namespace (optional)
            
        Returns:
            Dictionary with generated code and source references
        """
        print(f"🔍 Searching for relevant Terraform examples...")
        
        # Search for relevant Terraform code examples
        search_results = self.search_similar_code(request, top_k=top_k, namespace=namespace)
        
        if not search_results.matches:
            return {
                "request": request,
                "code": "# No relevant examples found in the database.\n# Unable to generate code.",
                "sources": [],
                "status": "no_examples"
            }
        
        print(f"✓ Found {len(search_results.matches)} relevant examples")
        print(f"🤖 Generating Terraform code...")
        
        # Generate Terraform code based on retrieved examples
        generated_code = self.generate_terraform_code(request, search_results.matches)
        
        # Format source references
        sources = [
            {
                "score": float(match.score),
                "id": match.id,
                "metadata_keys": list(match.metadata.keys())
            }
            for match in search_results.matches
        ]
        
        print(f"✓ Code generated successfully!")
        
        return {
            "request": request,
            "code": generated_code,
            "sources": sources,
            "status": "success",
            "num_examples_used": len(search_results.matches)
        }
    
    def batch_embed(self, texts):
        """
        Create embeddings for multiple texts (useful for indexing)
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        return embeddings.tolist()


# Alternative: Using more powerful models for better code generation
class TerraformCodeGeneratorAdvanced:
    def __init__(self, pinecone_api_key, embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 llm_model="bigcode/starcoder"):
        """
        Initialize with more advanced code generation model
        Note: Requires more GPU memory
        """
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = "terraform-iac-v1"
        self.index = self.pc.Index(self.index_name)
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Load advanced code generation model
        print(f"Loading Advanced Code Generation Model: {llm_model}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        print(f"Advanced models loaded on {self.device}")
    
    def create_embedding(self, text):
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def search_similar_code(self, query, top_k=5):
        query_embedding = self.create_embedding(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results
    
    def generate_terraform_code(self, query, context_docs):
        # Extract code examples
        code_examples = []
        for match in context_docs[:3]:
            code_keys = ['code', 'terraform_code', 'content', 'text']
            for key in code_keys:
                if key in match.metadata and match.metadata[key]:
                    code_examples.append(match.metadata[key])
                    break
        
        context = "\n\n".join([f"```hcl\n{code}\n```" for code in code_examples[:2]])
        
        prompt = f"""<|fim_prefix|>Generate Terraform code for: {query}

Reference examples:
{context}

# Solution:
<|fim_suffix|><|fim_middle|>"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=600,
                temperature=0.2,
                do_sample=True,
                top_p=0.95
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated code
        if "<|fim_middle|>" in generated_text:
            code = generated_text.split("<|fim_middle|>")[-1].split("<|fim_suffix|>")[0].strip()
        else:
            code = generated_text[len(prompt):].strip()
        
        return code
    
    def generate(self, request, top_k=5):
        search_results = self.search_similar_code(request, top_k=top_k)
        
        if not search_results.matches:
            return {
                "request": request,
                "code": "# No examples found",
                "sources": []
            }
        
        generated_code = self.generate_terraform_code(request, search_results.matches)
        
        sources = [
            {
                "score": float(match.score),
                "id": match.id
            }
            for match in search_results.matches
        ]
        
        return {
            "request": request,
            "code": generated_code,
            "sources": sources
        }


# Example usage
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    if not PINECONE_API_KEY:
        raise ValueError(
            "Pinecone API key not found!\n"
            "Please create a .env file with: PINECONE_API_KEY=your-key"
        )
    
    # Initialize Terraform Code Generator
    print("="*60)
    print("Initializing Terraform Code Generator with RAG")
    print("="*60)
    
    generator = TerraformCodeGenerator(
        pinecone_api_key=PINECONE_API_KEY,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="Salesforce/codegen-350M-mono"
    )
    
    # Get index statistics
    print("\n📊 Index Statistics:")
    stats = generator.get_index_stats()
    print(f"   Total vectors: {stats.total_vector_count}")
    print(f"   Dimension: {stats.dimension}")
    
    # Terraform code generation requests
    requests = [
        "Create an AWS EC2 instance with a security group that allows SSH and HTTP",
        "Generate S3 bucket with versioning and encryption enabled",
        "Create a VPC with public and private subnets",
        "Deploy an RDS MySQL database instance",
        "Create an AWS Lambda function with API Gateway"
    ]
    
    # Generate code for each request
    for request in requests:
        print("\n" + "="*60)
        print(f"📝 Request: {request}")
        print("="*60)
        
        result = generator.generate(request, top_k=5)
        
        print(f"\n🎯 Status: {result['status']}")
        print(f"📚 Examples used: {result.get('num_examples_used', 0)}")
        
        print("\n💻 Generated Terraform Code:")
        print("-" * 60)
        print(result['code'])
        print("-" * 60)
        
        print("\n📖 Source References:")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"   {i}. Similarity: {source['score']:.4f} | ID: {source['id']}")
        
        print()
    
    # Single code generation example
    print("\n" + "="*60)
    print("Single Code Generation Example")
    print("="*60)
    
    single_request = "Create an ECS cluster with Fargate task definition"
    result = generator.generate(single_request, top_k=3)
    
    print(f"\n💻 Generated Code:\n{result['code']}")
    
    # Uncomment to use advanced model (requires powerful GPU)
    # print("\n\nUsing Advanced Model (StarCoder)...")
    # advanced_generator = TerraformCodeGeneratorAdvanced(
    #     pinecone_api_key=PINECONE_API_KEY
    # )
    # result_advanced = advanced_generator.generate(single_request, top_k=3)
    # print(result_advanced['code'])