import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Initialize API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Validate API keys
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found. Please set it in your .env file or environment variables.")
if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY not found. Please set it in your .env file or environment variables.")

class DocumentationScraper:
    def __init__(self):
        self.visited_urls = set()
        self.documents = []
        
    def scrape_terraform_docs(self, max_pages=100):
        """Scrape Terraform AWS Provider documentation"""
        base_url = "https://registry.terraform.io/providers/hashicorp/aws/latest/docs"
        
        print("Scraping Terraform AWS Provider documentation...")
        
        # Key resource pages to scrape
        resource_categories = [
            "/resources/vpc",
            "/resources/subnet",
            "/resources/instance",
            "/resources/s3_bucket",
            "/resources/security_group",
            "/resources/iam_role",
            "/resources/iam_policy",
            "/resources/rds_instance",
            "/resources/ecs_cluster",
            "/resources/ecs_service",
            "/resources/lambda_function",
            "/resources/api_gateway_rest_api",
            "/resources/cloudwatch_log_group",
            "/resources/route53_zone",
            "/resources/alb",
            "/resources/autoscaling_group",
            "/data-sources/ami",
            "/data-sources/availability_zones",
        ]
        
        for resource_path in resource_categories[:max_pages]:
            url = base_url + resource_path
            self._scrape_page(url, "terraform")
            time.sleep(1)  # Be respectful to the server
            
        print(f"✓ Scraped {len([d for d in self.documents if d.metadata['source_type'] == 'terraform'])} Terraform pages")
        
    def scrape_aws_docs(self, max_pages=50):
        """Scrape AWS documentation for key services"""
        print("Scraping AWS documentation...")
        
        # Key AWS documentation pages
        aws_docs = [
            "https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html",
            "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html",
            "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
            "https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html",
            "https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Welcome.html",
            "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
            "https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html",
        ]
        
        for url in aws_docs[:max_pages]:
            self._scrape_page(url, "aws")
            time.sleep(1)
            
        print(f"✓ Scraped {len([d for d in self.documents if d.metadata['source_type'] == 'aws'])} AWS pages")
        
    def scrape_terraform_examples(self):
        """Scrape Terraform example configurations from GitHub"""
        print("Fetching Terraform examples from GitHub...")
        
        # Popular Terraform AWS examples
        github_examples = [
            "https://raw.githubusercontent.com/hashicorp/terraform-provider-aws/main/examples/vpc/main.tf",
            "https://raw.githubusercontent.com/terraform-aws-modules/terraform-aws-vpc/master/examples/complete/main.tf",
            "https://raw.githubusercontent.com/terraform-aws-modules/terraform-aws-ec2-instance/master/examples/complete/main.tf",
            "https://raw.githubusercontent.com/terraform-aws-modules/terraform-aws-s3-bucket/master/examples/complete/main.tf",
        ]
        
        for url in github_examples:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    doc = Document(
                        page_content=response.text,
                        metadata={
                            "source": url,
                            "source_type": "example",
                            "type": "terraform_code"
                        }
                    )
                    self.documents.append(doc)
                    print(f"✓ Fetched example: {url.split('/')[-1]}")
                time.sleep(1)
            except Exception as e:
                print(f"✗ Failed to fetch {url}: {str(e)}")
                
    def _scrape_page(self, url, source_type):
        """Scrape a single page"""
        if url in self.visited_urls:
            return
            
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text(separator='\n', strip=True)
                
                # Extract code examples if present
                code_blocks = soup.find_all(['code', 'pre'])
                code_examples = '\n\n'.join([block.get_text() for block in code_blocks])
                
                if code_examples:
                    text += f"\n\nCode Examples:\n{code_examples}"
                
                if len(text) > 100:  # Only save if substantial content
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": url,
                            "source_type": source_type,
                            "title": soup.title.string if soup.title else ""
                        }
                    )
                    self.documents.append(doc)
                    self.visited_urls.add(url)
                    print(f"✓ Scraped: {url}")
                    
        except Exception as e:
            print(f"✗ Failed to scrape {url}: {str(e)}")
            
    def get_documents(self):
        """Return all scraped documents"""
        return self.documents


class TerraformRAGSystem:
    def __init__(self, index_name="terraform-iac"):
        self.index_name = index_name
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        
    def initialize_embeddings(self):
        """Initialize HuggingFace embeddings"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("✓ Embeddings initialized")
        
    def initialize_pinecone(self):
        """Initialize Pinecone vector database"""
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Create index if it doesn't exist
        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"✓ Created Pinecone index: {self.index_name}")
        else:
            print(f"✓ Using existing Pinecone index: {self.index_name}")
            
    def process_documents(self, documents):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        print(f"✓ Processed {len(documents)} documents into {len(splits)} chunks")
        return splits
        
    def create_vectorstore(self, documents):
        """Create vector store from documents"""
        self.vectorstore = LangchainPinecone.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.index_name
        )
        print("✓ Vector store created and documents indexed")
        
    def initialize_llm(self, use_local=False):
        """Initialize HuggingFace LLM or local Ollama model"""
        if use_local:
            # Use Ollama for local inference
            try:
                from langchain_community.llms import Ollama
                self.llm = Ollama(
                    model="codellama:13b",
                    temperature=0.3
                )
                print(f"✓ LLM initialized with local Ollama: codellama:13b")
                return
            except Exception as e:
                print(f"✗ Ollama not available: {e}")
                print("Falling back to HuggingFace API...")
        
        # Updated list of working models (as of 2024)
        # These models are known to work well with HuggingFace Inference API
        models_to_try = [
            {
                "repo_id": "mistralai/Mistral-7B-Instruct-v0.2",
                "max_tokens": 4096,
                "description": "Mistral 7B Instruct (recommended)"
            },
            {
                "repo_id": "HuggingFaceH4/zephyr-7b-beta",
                "max_tokens": 2048,
                "description": "Zephyr 7B Beta"
            },
            {
                "repo_id": "meta-llama/Llama-2-7b-chat-hf",
                "max_tokens": 2048,
                "description": "Llama 2 7B Chat"
            },
            {
                "repo_id": "google/flan-t5-xxl",
                "max_tokens": 1024,
                "description": "FLAN-T5 XXL"
            },
            {
                "repo_id": "bigscience/bloom-7b1",
                "max_tokens": 1024,
                "description": "BLOOM 7B"
            }
        ]
        
        for model_config in models_to_try:
            try:
                print(f"Attempting to initialize: {model_config['description']}...")
                self.llm = HuggingFaceEndpoint(
                    repo_id=model_config["repo_id"],
                    temperature=0.3,
                    max_new_tokens=model_config["max_tokens"],
                    top_p=0.95,
                    huggingfacehub_api_token=HUGGINGFACE_API_KEY
                )
                print(f"✓ LLM initialized: {model_config['description']}")
                return
            except Exception as e:
                print(f"✗ {model_config['description']} failed: {str(e)}")
                continue
        
        # If all models fail, raise an error with helpful message
        raise Exception(
            "All model initialization attempts failed. Please check:\n"
            "1. Your HuggingFace API key is valid\n"
            "2. You have accepted model licenses on HuggingFace (especially for Llama models)\n"
            "3. The HuggingFace Inference API is operational\n"
            "4. Consider using use_local=True to run with Ollama instead"
        )
        
    def create_qa_chain(self):
        """Create QA chain with custom prompt"""
        template = """You are an expert AWS Terraform developer. Use the following Terraform documentation and AWS service information to generate high-quality Infrastructure as Code.

Context from documentation:
{context}

User Request: {question}

Instructions:
1. Generate valid, production-ready Terraform code for AWS
2. Follow AWS and Terraform best practices
3. Include necessary provider configuration
4. Add meaningful comments explaining the code
5. Use appropriate resource naming conventions
6. Include variables where appropriate
7. Add outputs for important resource attributes
8. Consider security best practices (IAM, security groups, encryption)

Generate the complete Terraform configuration:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        print("✓ QA chain created")
        
    def generate_terraform_code(self, query):
        """Generate Terraform code based on user query"""
        result = self.qa_chain.invoke({"query": query})
        return {
            "code": result["result"],
            "sources": result["source_documents"]
        }
        
    def setup_from_web(self, scrape_terraform=True, scrape_aws=True, scrape_examples=True, use_local=False):
        """Complete setup process by scraping web documentation"""
        print("=" * 60)
        print("Setting up Terraform RAG System with Web Scraping")
        print("=" * 60)
        
        # Scrape documentation
        scraper = DocumentationScraper()
        
        if scrape_terraform:
            scraper.scrape_terraform_docs(max_pages=50)
            
        if scrape_aws:
            scraper.scrape_aws_docs(max_pages=20)
            
        if scrape_examples:
            scraper.scrape_terraform_examples()
            
        documents = scraper.get_documents()
        print(f"\n✓ Total documents collected: {len(documents)}")
        
        # Initialize and build RAG system
        self.initialize_embeddings()
        self.initialize_pinecone()
        processed_docs = self.process_documents(documents)
        self.create_vectorstore(processed_docs)
        self.initialize_llm(use_local=use_local)
        self.create_qa_chain()
        
        print("\n" + "=" * 60)
        print("✓ RAG System is ready to generate Terraform code!")
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag_system = TerraformRAGSystem(index_name="terraform-iac-v1")
    
    # Setup by scraping documentation from the web
    # Set use_local=True if you want to use Ollama instead of HuggingFace API
    rag_system.setup_from_web(
        scrape_terraform=True,
        scrape_aws=True,
        scrape_examples=True,
        use_local=False  # Change to True to use Ollama
    )
    
    # Example queries
    example_queries = [
        "Create an AWS VPC with public and private subnets across 3 availability zones",
        "Create an S3 bucket with versioning and encryption enabled",
        "Deploy an EC2 instance with a security group allowing SSH access",
        "Create an RDS PostgreSQL database with automated backups"
    ]
    
    # Generate code for first example
    print("\n\nExample Generation:")
    print("=" * 60)
    result = rag_system.generate_terraform_code(example_queries[0])
    print("\nGenerated Terraform Code:")
    print(result["code"])
    print("\n\nSources Used:")
    for i, doc in enumerate(result["sources"], 1):
        print(f"{i}. {doc.metadata.get('source_type', 'Unknown')}: {doc.metadata.get('source', 'N/A')[:80]}")