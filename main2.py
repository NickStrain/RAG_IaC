# import ollama

# code_prompt = "Write a Python function that checks if a number is prime."
# response = ollama.generate(model='codellama:7b', prompt=code_prompt)
# print(response['response'])

import ollama 
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv  
load_dotenv()  # Load environment variables from .env file

class RAGSystem:
    def __init__(self, pinecone_api_key, index_name="terraform-iac-v1", 
                 generation_model="codellama:7b"):
        """
        Initialize RAG system with Pinecone and Ollama
        
        Args:
            pinecone_api_key: Your Pinecone API key
            index_name: Name of your Pinecone index
            generation_model: Ollama model to use for text generation
        """
        self.generation_model = generation_model
        
        # Initialize SentenceTransformer for embeddings (384 dimensions)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        self.index = pc.Index(index_name)
        
        print(f"Connected to Pinecone index: {index_name}")
        print(f"Using embedding model: all-MiniLM-L6-v2 (SentenceTransformer)")
        print(f"Using generation model: {generation_model}")
    
    def get_embedding(self, text):
        """
        Generate embeddings using SentenceTransformer
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding (384 dimensions)
        """
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def retrieve_documents(self, query, top_k=3):
        """
        Retrieve relevant documents from Pinecone
        
        Args:
            query: User query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        # Generate embedding for the query
        query_embedding = self.get_embedding(query)
        
        # Search Pinecone index
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results['matches']
    
    def format_context(self, documents):
        """
        Format retrieved documents into context string
        
        Args:
            documents: List of documents from Pinecone
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.get('metadata', {})
            text = metadata.get('text', 'No content available')
            score = doc.get('score', 0)
            
            context_parts.append(f"[Document {i}] (Relevance: {score:.3f})\n{text}")
        
        return "\n\n".join(context_parts)
    
    def generate_response(self, query, context):
        """
        Generate response using Ollama with retrieved context
        
        Args:
            query: User query
            context: Retrieved context from Pinecone
            
        Returns:
            Generated response text
        """
        # Create augmented prompt with context
        augmented_prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:

Note:
just give the terraform code not need of explanation
try to include industry best practices in the code
use the context as an reference.
make sure the code is correct and complete.


"""
        
        # Generate response using Ollama
        # response = ollama.generate(
        #     model=self.generation_model,
        #     prompt=augmented_prompt
        # )
        response = ollama.generate(model='codellama:7b-instruct',prompt=augmented_prompt)
        
        return response['response']
    
    def query(self, user_query, top_k=3, verbose=True):
        """
        Complete RAG pipeline: retrieve and generate
        
        Args:
            user_query: User's question
            top_k: Number of documents to retrieve
            verbose: Print intermediate steps
            
        Returns:
            Dictionary with answer and retrieved documents
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Query: {user_query}")
            print(f"{'='*60}\n")
        
        # Step 1: Retrieve relevant documents
        if verbose:
            print("🔍 Retrieving relevant documents...")
        
        documents = self.retrieve_documents(user_query, top_k=top_k)
        
        if verbose:
            print(f"✓ Retrieved {len(documents)} documents\n")
        
        # Step 2: Format context
        context = self.format_context(documents)
        
        if verbose:
            print("📄 Context:")
            print("-" * 60)
            print(context[:500] + "..." if len(context) > 500 else context)
            print("-" * 60 + "\n")
        
        # Step 3: Generate response
        if verbose:
            print("🤖 Generating response...\n")
        
        answer = self.generate_response(user_query, context)
        
        if verbose:
            print("💡 Answer:")
            print("-" * 60)
            print(answer)
            print("-" * 60)
        
        return {
            'answer': answer,
            'documents': documents,
            'context': context
        }


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-api-key-here")
    
    rag = RAGSystem(
        pinecone_api_key=PINECONE_API_KEY,
        index_name="terraform-iac-v1",
        generation_model="codellama:7b"
    )
    
    # Example queries
    queries = [
        "How do I create an S3 bucket in Terraform?"
       
    ]
    
    # Run RAG for each query
    for query in queries:
        result = rag.query(query, top_k=3, verbose=True)
        print("\n" + "="*60 + "\n")