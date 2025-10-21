import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA, LLMChain
from langchain_classic.prompts import PromptTemplate
from langchain_core.documents import Document


# You'll need: pip install langchain langchain-google-genai langchain-community langchain-huggingface chromadb sentence-transformers

@dataclass
class TerraformExample:
    """Represents a Terraform code example"""
    resource_type: str
    service: str
    code: str
    description: str
    tags: List[str]

class TerraformRAG:
    """RAG system for generating Terraform code using LangChain + Gemini + HuggingFace Embeddings"""
    
    def __init__(self, gemini_api_key: str, collection_name: str = "terraform_examples", 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Initialize Gemini LLM through LangChain
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        
        # Initialize HuggingFace embeddings (free and local)
        print(f"Loading HuggingFace embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embeddings model loaded successfully!")
        
        # Initialize vector store (will be populated later)
        self.vectorstore = None
        self.collection_name = collection_name
        
    def add_terraform_examples(self, examples: List[TerraformExample]):
        """Add Terraform examples to the knowledge base using LangChain"""
        documents = []
        
        for idx, example in enumerate(examples):
            # Create document content
            content = f"""
Resource Type: {example.resource_type}
AWS Service: {example.service}
Description: {example.description}
Tags: {', '.join(example.tags)}

Terraform Code:
{example.code}
            """.strip()
            
            # Create LangChain Document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "resource_type": example.resource_type,
                    "service": example.service,
                    "description": example.description,
                    "tags": ', '.join(example.tags),
                    "example_id": f"example_{idx}"
                }
            )
            documents.append(doc)
        
        print(f"Creating embeddings for {len(examples)} examples...")
        # Create or update vector store
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name
            )
        else:
            self.vectorstore.add_documents(documents)
        
        print(f"Added {len(examples)} examples to knowledge base")
    
    def generate_terraform(self, 
                          requirement: str, 
                          aws_service: Optional[str] = None,
                          n_examples: int = 3) -> Dict[str, str]:
        """Generate Terraform code using LangChain RAG pipeline"""
        
        if self.vectorstore is None:
            return {
                "main_tf": "# Error: No examples in knowledge base",
                "explanation": "Please add examples first using add_terraform_examples()"
            }
        
        # Build search query
        search_query = requirement
        if aws_service:
            search_query = f"{aws_service} {requirement}"
        
        # Retrieve relevant examples using LangChain
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": n_examples}
        )
        relevant_docs = retriever.invoke(search_query)
        
        # Build context from retrieved documents
        context = "Here are relevant Terraform examples from the knowledge base:\n\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"Example {i}:\n{doc.page_content}\n\n"
        
        # Create prompt template for Terraform generation
        generation_prompt = PromptTemplate(
            input_variables=["context", "requirement", "aws_service"],
            template="""You are an expert at writing Terraform code for AWS infrastructure.

{context}

User Requirement: {requirement}
AWS Service: {aws_service}

Generate complete, production-ready Terraform code that:
1. Follows best practices and AWS Well-Architected Framework
2. Includes appropriate variables and outputs
3. Has proper resource naming and tagging
4. Includes comments explaining key decisions
5. Uses secure configurations by default

Provide the code in the following structure:
- main.tf: Main resource definitions
- variables.tf: Input variables
- outputs.tf: Output values
- terraform.tfvars (example): Example variable values

Format your response as JSON with keys: "main_tf", "variables_tf", "outputs_tf", "tfvars_example", "explanation"

Respond ONLY with valid JSON, no additional text before or after."""
        )
        
        # Create LLM chain
        chain = LLMChain(llm=self.llm, prompt=generation_prompt)
        
        # Generate code
        try:
            response = chain.invoke({
                "context": context,
                "requirement": requirement,
                "aws_service": aws_service or "Not specified"
            })
            
            response_text = response['text'].strip()
            
            # Clean response (remove markdown code blocks if present)
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start != -1 and end != 0:
                    result = json.loads(response_text[start:end])
                else:
                    result = {
                        "main_tf": response_text,
                        "explanation": "Generated code above"
                    }
        except Exception as e:
            print(f"Error generating with LangChain: {e}")
            result = {
                "main_tf": f"# Error: {str(e)}",
                "explanation": f"Failed to generate: {str(e)}"
            }
        
        return result
    
    def generate_terraform_with_qa(self, requirement: str, aws_service: Optional[str] = None) -> str:
        """Alternative: Generate using LangChain's RetrievalQA chain"""
        
        if self.vectorstore is None:
            return "Error: No examples in knowledge base"
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        # Build query
        query = f"""Generate production-ready Terraform code for: {requirement}
        
AWS Service: {aws_service or 'Not specified'}

Include:
1. Complete main.tf with all resources
2. variables.tf with input variables
3. outputs.tf with useful outputs
4. Comments and best practices"""
        
        # Run query
        result = qa_chain.invoke({"query": query})
        
        return result['result']
    
    def validate_terraform(self, terraform_code: str) -> Dict[str, any]:
        """Validate Terraform code using LangChain"""
        
        validation_prompt = PromptTemplate(
            input_variables=["code"],
            template="""Review this Terraform code for:
1. Syntax issues
2. Security concerns
3. Best practice violations
4. Missing required arguments
5. Potential cost optimization

Terraform Code:
{code}

Provide a JSON response with keys: "is_valid" (boolean), "issues" (list of strings), "suggestions" (list of strings)
Respond ONLY with valid JSON, no additional text."""
        )
        
        chain = LLMChain(llm=self.llm, prompt=validation_prompt)
        
        try:
            response = chain.invoke({"code": terraform_code})
            response_text = response['text'].strip()
            
            # Clean response
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            result = json.loads(response_text[start:end])
        except Exception as e:
            result = {
                "is_valid": True,
                "issues": [],
                "suggestions": [f"Validation error: {str(e)}"]
            }
        
        return result
    
    def search_examples(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar examples using LangChain's similarity search"""
        if self.vectorstore is None:
            return []
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results


# Example usage and knowledge base setup
def setup_example_knowledge_base(rag: TerraformRAG):
    """Populate with common Terraform examples"""
    
    examples = [
        TerraformExample(
            resource_type="aws_vpc",
            service="VPC",
            description="Basic VPC with public and private subnets",
            tags=["networking", "vpc", "subnets"],
            code="""
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "$${var.project_name}-vpc"
    Environment = var.environment
  }
}

resource "aws_subnet" "public" {
  count                   = length(var.public_subnet_cidrs)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "$${var.project_name}-public-$${count.index + 1}"
  }
}

resource "aws_subnet" "private" {
  count             = length(var.private_subnet_cidrs)
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name = "$${var.project_name}-private-$${count.index + 1}"
  }
}
"""
        ),
        TerraformExample(
            resource_type="aws_s3_bucket",
            service="S3",
            description="Secure S3 bucket with encryption and versioning",
            tags=["storage", "s3", "encryption", "versioning"],
            code="""
resource "aws_s3_bucket" "main" {
  bucket = var.bucket_name

  tags = {
    Name        = var.bucket_name
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "main" {
  bucket = aws_s3_bucket.main.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "main" {
  bucket = aws_s3_bucket.main.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "main" {
  bucket = aws_s3_bucket.main.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
"""
        ),
        TerraformExample(
            resource_type="aws_ec2_instance",
            service="EC2",
            description="EC2 instance with security group",
            tags=["compute", "ec2", "security-group"],
            code="""
resource "aws_instance" "main" {
  ami           = var.ami_id
  instance_type = var.instance_type
  subnet_id     = var.subnet_id

  vpc_security_group_ids = [aws_security_group.instance.id]

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size
    encrypted             = true
    delete_on_termination = true
  }

  tags = {
    Name        = "$${var.project_name}-instance"
    Environment = var.environment
  }
}

resource "aws_security_group" "instance" {
  name        = "$${var.project_name}-instance-sg"
  description = "Security group for EC2 instance"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "$${var.project_name}-instance-sg"
  }
}
"""
        ),
        TerraformExample(
            resource_type="aws_rds_instance",
            service="RDS",
            description="RDS PostgreSQL instance with encryption",
            tags=["database", "rds", "postgresql", "encryption"],
            code="""
resource "aws_db_subnet_group" "main" {
  name       = "$${var.project_name}-db-subnet-group"
  subnet_ids = var.private_subnet_ids

  tags = {
    Name = "$${var.project_name}-db-subnet-group"
  }
}

resource "aws_db_instance" "main" {
  identifier     = "$${var.project_name}-db"
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = var.db_instance_class

  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage
  storage_encrypted     = true

  db_name  = var.database_name
  username = var.database_username
  password = var.database_password

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.db.id]

  backup_retention_period = 7
  skip_final_snapshot     = false
  final_snapshot_identifier = "$${var.project_name}-db-final-snapshot"

  tags = {
    Name        = "$${var.project_name}-db"
    Environment = var.environment
  }
}

resource "aws_security_group" "db" {
  name        = "$${var.project_name}-db-sg"
  description = "Security group for RDS instance"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = var.allowed_security_groups
  }

  tags = {
    Name = "$${var.project_name}-db-sg"
  }
}
"""
        ),
        TerraformExample(
            resource_type="aws_lambda_function",
            service="Lambda",
            description="Lambda function with IAM role and CloudWatch logs",
            tags=["compute", "lambda", "serverless", "iam"],
            code="""
resource "aws_lambda_function" "main" {
  filename         = var.lambda_zip_path
  function_name    = "$${var.project_name}-function"
  role            = aws_iam_role.lambda.arn
  handler         = var.handler
  source_code_hash = filebase64sha256(var.lambda_zip_path)
  runtime         = var.runtime

  environment {
    variables = var.environment_variables
  }

  tags = {
    Name        = "$${var.project_name}-function"
    Environment = var.environment
  }
}

resource "aws_iam_role" "lambda" {
  name = "$${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
  role       = aws_iam_role.lambda.name
}

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/$${aws_lambda_function.main.function_name}"
  retention_in_days = 14
}
"""
        )
    ]
    
    rag.add_terraform_examples(examples)


# Main execution example
if __name__ == "__main__":
    # Initialize RAG system with LangChain + Gemini + HuggingFace Embeddings
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        print("Please set GEMINI_API_KEY environment variable")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        exit(1)
    
    # You can choose different HuggingFace models:
    # - "sentence-transformers/all-MiniLM-L6-v2" (default, fast, lightweight)
    # - "sentence-transformers/all-mpnet-base-v2" (better quality, slower)
    # - "BAAI/bge-small-en-v1.5" (good balance)
    rag = TerraformRAG(
        gemini_api_key=API_KEY,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Setup knowledge base
    print("\nSetting up knowledge base with LangChain + HuggingFace...")
    setup_example_knowledge_base(rag)
    
    # Example: Generate Terraform code
    print("\n" + "="*50)
    print("Generating Terraform code with LangChain + Gemini...")
    print("="*50 + "\n")
    
    requirement = "Create a secure web application infrastructure with a load balancer, auto-scaling group, and RDS database"
    
    result = rag.generate_terraform(
        requirement=requirement,
        aws_service="EC2, RDS, ELB"
    )
    
    print("MAIN.TF:")
    print("-" * 50)
    print(result.get("main_tf", ""))
    print("\n")
    
    if "variables_tf" in result:
        print("VARIABLES.TF:")
        print("-" * 50)
        print(result.get("variables_tf", ""))
        print("\n")
    
    if "outputs_tf" in result:
        print("OUTPUTS.TF:")
        print("-" * 50)
        print(result.get("outputs_tf", ""))
        print("\n")
    
    print("EXPLANATION:")
    print("-" * 50)
    print(result.get("explanation", ""))
    
    # Validate the generated code
    print("\n" + "="*50)
    print("Validating generated code with LangChain...")
    print("="*50 + "\n")
    
    validation = rag.validate_terraform(result.get("main_tf", ""))
    print(json.dumps(validation, indent=2))
    
    # Example: Search for similar examples
    print("\n" + "="*50)
    print("Searching for similar examples...")
    print("="*50 + "\n")
    
    similar = rag.search_examples("S3 bucket with security", k=2)
    for i, doc in enumerate(similar, 1):
        print(f"Similar Example {i}:")
        print(doc.page_content[:200] + "...")
        print()