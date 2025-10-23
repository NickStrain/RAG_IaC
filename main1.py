import os
import json
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import re
import time

load_dotenv()

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import LLMChain
from langchain_classic.prompts import PromptTemplate
from langchain_core.documents import Document

@dataclass
class TerraformExample:
    """Represents a Terraform code example"""
    resource_type: str
    service: str
    code: str
    description: str
    tags: List[str]
    source_url: str = ""


class ImprovedTerraformScraper:
    """Improved scraper with better error handling and alternative sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.github_token = os.getenv("GITHUB_TOKEN")
        
    def fetch_from_terraform_aws_modules(self, module_name: str) -> List[TerraformExample]:
        """Fetch examples from terraform-aws-modules GitHub (more reliable)"""
        examples = []
        
        repos = {
            "vpc": "terraform-aws-modules/terraform-aws-vpc",
            "s3": "terraform-aws-modules/terraform-aws-s3-bucket",
            "ec2": "terraform-aws-modules/terraform-aws-ec2-instance",
            "rds": "terraform-aws-modules/terraform-aws-rds",
            "lambda": "terraform-aws-modules/terraform-aws-lambda",
            "alb": "terraform-aws-modules/terraform-aws-alb",
            "ecs": "terraform-aws-modules/terraform-aws-ecs",
            "eks": "terraform-aws-modules/terraform-aws-eks",
            "security-group": "terraform-aws-modules/terraform-aws-security-group"
        }
        
        repo = repos.get(module_name.lower())
        if not repo:
            print(f"No known module for {module_name}")
            return examples
            
        try:
            # Try multiple files
            files_to_check = [
                "README.md",
                "examples/complete/main.tf",
                "examples/simple/main.tf"
            ]
            
            for file_path in files_to_check:
                url = f"https://raw.githubusercontent.com/{repo}/master/{file_path}"
                
                print(f"  Trying: {url}")
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    content = response.text
                    
                    # Extract HCL code blocks
                    if file_path.endswith('.md'):
                        code_blocks = re.findall(r'```(?:hcl|terraform)?\n(.*?)```', content, re.DOTALL)
                    else:
                        code_blocks = [content]
                    
                    for idx, code in enumerate(code_blocks):
                        if len(code) > 50 and ('resource' in code or 'module' in code):
                            examples.append(TerraformExample(
                                resource_type=f"aws_{module_name}",
                                service=module_name.upper(),
                                code=code[:3000],  # Limit size
                                description=f"{module_name} example from {file_path}",
                                tags=["terraform-aws-modules", module_name, "official"],
                                source_url=f"https://github.com/{repo}/blob/master/{file_path}"
                            ))
                    
                    if examples:
                        print(f"  ✓ Found {len(code_blocks)} examples in {file_path}")
                        break
                        
                time.sleep(0.5)  # Rate limiting
                
        except Exception as e:
            print(f"  ✗ Error fetching {module_name}: {e}")
        
        return examples
    
    def fetch_from_github_api(self, query: str, max_results: int = 3) -> List[TerraformExample]:
        """Fetch using GitHub API with better error handling"""
        examples = []
        
        try:
            url = "https://api.github.com/search/code"
            headers = {'Accept': 'application/vnd.github+json'}
            
            if self.github_token:
                headers['Authorization'] = f'Bearer {self.github_token}'
                print("  Using authenticated GitHub API")
            else:
                print("  ⚠ Using unauthenticated GitHub API (limited to 10 requests/minute)")
            
            params = {
                'q': f'{query} language:HCL path:/ filename:main.tf',
                'per_page': max_results
            }
            
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 403:
                print("  ✗ GitHub API rate limit exceeded")
                return examples
            elif response.status_code != 200:
                print(f"  ✗ GitHub API error: {response.status_code}")
                return examples
            
            data = response.json()
            items = data.get('items', [])
            
            print(f"  Found {len(items)} results")
            
            for item in items[:max_results]:
                try:
                    # Use API to get content
                    content_url = item.get('url')
                    content_response = self.session.get(content_url, headers=headers, timeout=10)
                    
                    if content_response.status_code == 200:
                        import base64
                        content_data = content_response.json()
                        code = base64.b64decode(content_data['content']).decode('utf-8')
                        
                        # Extract resource type
                        resource_match = re.search(r'resource\s+"(aws_\w+)"\s+"(\w+)"', code)
                        if resource_match:
                            resource_type = resource_match.group(1)
                            
                            examples.append(TerraformExample(
                                resource_type=resource_type,
                                service=resource_type.split('_')[1].upper(),
                                code=code[:2000],
                                description=f"Example from {item['repository']['full_name']}",
                                tags=["github", resource_type, "community"],
                                source_url=item['html_url']
                            ))
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    print(f"  ✗ Error processing item: {e}")
                    
        except Exception as e:
            print(f"  ✗ GitHub search error: {e}")
        
        return examples
    
    def create_synthetic_examples(self, service: str) -> List[TerraformExample]:
        """Create basic template examples as fallback"""
        templates = {
            "vpc": '''resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-vpc"
  }
}

resource "aws_subnet" "public" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name = "${var.project_name}-public-${count.index + 1}"
  }
}''',
            "s3": '''resource "aws_s3_bucket" "main" {
  bucket = var.bucket_name

  tags = {
    Name = var.bucket_name
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
}''',
            "ec2": '''resource "aws_instance" "main" {
  ami           = var.ami_id
  instance_type = var.instance_type
  subnet_id     = var.subnet_id

  vpc_security_group_ids = [aws_security_group.main.id]

  root_block_device {
    volume_size = var.root_volume_size
    encrypted   = true
  }

  tags = {
    Name = "${var.project_name}-instance"
  }
}

resource "aws_security_group" "main" {
  name_prefix = "${var.project_name}-"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.admin_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}''',
            "rds": '''resource "aws_db_instance" "main" {
  identifier             = var.db_identifier
  engine                 = var.db_engine
  engine_version         = var.db_engine_version
  instance_class         = var.db_instance_class
  allocated_storage      = var.db_allocated_storage
  storage_encrypted      = true
  
  db_name  = var.db_name
  username = var.db_username
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.db.id]

  backup_retention_period = 7
  skip_final_snapshot     = false
  final_snapshot_identifier = "${var.db_identifier}-final"

  tags = {
    Name = var.db_identifier
  }
}''',
            "lambda": '''resource "aws_lambda_function" "main" {
  filename      = var.lambda_zip_path
  function_name = var.function_name
  role          = aws_iam_role.lambda.arn
  handler       = var.handler
  runtime       = var.runtime

  source_code_hash = filebase64sha256(var.lambda_zip_path)

  environment {
    variables = var.environment_variables
  }

  tags = {
    Name = var.function_name
  }
}

resource "aws_iam_role" "lambda" {
  name = "${var.function_name}-role"

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
}'''
        }
        
        examples = []
        if service.lower() in templates:
            examples.append(TerraformExample(
                resource_type=f"aws_{service.lower()}",
                service=service.upper(),
                code=templates[service.lower()],
                description=f"Template example for {service}",
                tags=["template", service.lower(), "baseline"],
                source_url="built-in"
            ))
        
        return examples


class TerraformRAG:
    """RAG system for generating Terraform code"""
    
    def __init__(self, gemini_api_key: str, collection_name: str = "terraform_examples", 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 persist_directory: str = "./chroma_db"):
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        
        print(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Load existing or create new
        if os.path.exists(persist_directory):
            try:
                self.vectorstore = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=persist_directory
                )
                count = self.vectorstore._collection.count()
                print(f"✓ Loaded {count} existing examples")
            except Exception as e:
                print(f"Creating new vector store: {e}")
                self.vectorstore = None
        else:
            self.vectorstore = None
        
        self.scraper = ImprovedTerraformScraper()
    
    def fetch_examples_improved(self, services: List[str], use_templates: bool = True):
        """Improved fetching with multiple strategies"""
        all_examples = []
        
        print(f"\n{'='*70}")
        print("FETCHING TERRAFORM EXAMPLES")
        print(f"{'='*70}\n")
        
        for service in services:
            print(f"\n📦 Fetching {service.upper()} examples:")
            service_examples = []
            
            # Strategy 1: Official terraform-aws-modules
            print("  [1/3] Trying terraform-aws-modules...")
            module_examples = self.scraper.fetch_from_terraform_aws_modules(service)
            service_examples.extend(module_examples)
            
            # Strategy 2: GitHub search (if token available)
            if self.scraper.github_token:
                print("  [2/3] Searching GitHub...")
                github_examples = self.scraper.fetch_from_github_api(f"aws {service}", max_results=2)
                service_examples.extend(github_examples)
            else:
                print("  [2/3] Skipping GitHub search (no token)")
            
            # Strategy 3: Template fallback
            if use_templates and len(service_examples) < 2:
                print("  [3/3] Adding template examples...")
                template_examples = self.scraper.create_synthetic_examples(service)
                service_examples.extend(template_examples)
            
            print(f"  ✓ Total for {service}: {len(service_examples)} examples\n")
            all_examples.extend(service_examples)
        
        print(f"\n{'='*70}")
        print(f"TOTAL EXAMPLES FETCHED: {len(all_examples)}")
        print(f"{'='*70}\n")
        
        if all_examples:
            self.add_terraform_examples(all_examples)
        else:
            print("⚠ No examples found. Check your configuration.")
        
        return all_examples
    
    def add_terraform_examples(self, examples: List[TerraformExample]):
        """Add examples to vector store"""
        documents = []
        
        for idx, example in enumerate(examples):
            content = f"""
Resource Type: {example.resource_type}
AWS Service: {example.service}
Description: {example.description}
Tags: {', '.join(example.tags)}
Source: {example.source_url}

Terraform Code:
{example.code}
            """.strip()
            
            doc = Document(
                page_content=content,
                metadata={
                    "resource_type": example.resource_type,
                    "service": example.service,
                    "description": example.description,
                    "tags": ', '.join(example.tags),
                    "source_url": example.source_url,
                    "example_id": f"example_{idx}"
                }
            )
            documents.append(doc)
        
        print(f"Creating embeddings for {len(documents)} examples...")
        
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(documents)
        
        print(f"✓ Knowledge base updated: {self.vectorstore._collection.count()} total examples")
    
    def generate_terraform(self, requirement: str, aws_service: Optional[str] = None, 
                          n_examples: int = 3) -> Dict[str, str]:
        """Generate Terraform code using RAG"""
        
        if self.vectorstore is None or self.vectorstore._collection.count() == 0:
            return {
                "main_tf": "# Error: No examples in knowledge base",
                "explanation": "Please fetch examples first"
            }
        
        search_query = f"{aws_service} {requirement}" if aws_service else requirement
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": n_examples})
        relevant_docs = retriever.invoke(search_query)
        
        context = "Relevant Terraform examples:\n\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"Example {i}:\n{doc.page_content}\n\n"
        
        prompt = PromptTemplate(
            input_variables=["context", "requirement", "aws_service"],
            template="""You are a Terraform expert. Generate production-ready AWS infrastructure code.

{context}

Requirement: {requirement}
AWS Service: {aws_service}

Create complete Terraform code with:
1. main.tf - Resource definitions
2. variables.tf - Input variables  
3. outputs.tf - Output values
4. Brief explanation

Return valid JSON with keys: "main_tf", "variables_tf", "outputs_tf", "explanation"
JSON only, no markdown."""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            response = chain.invoke({
                "context": context,
                "requirement": requirement,
                "aws_service": aws_service or "Not specified"
            })
            
            text = response['text'].strip()
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'^```\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            
            result = json.loads(text)
            
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                result = {"main_tf": text, "explanation": "Generated code"}
        except Exception as e:
            result = {"main_tf": f"# Error: {e}", "explanation": str(e)}
        
        return result


# Main execution
if __name__ == "__main__":
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        print("❌ Please set GEMINI_API_KEY environment variable")
        exit(1)
    
    # Optional: Set GITHUB_TOKEN for better rate limits
    if not os.getenv("GITHUB_TOKEN"):
        print("ℹ️  Tip: Set GITHUB_TOKEN for better GitHub API access")
    
    rag = TerraformRAG(gemini_api_key=API_KEY)
    
    # Fetch examples with improved strategy
    services = ["vpc", "s3", "ec2", "rds", "lambda"]
    rag.fetch_examples_improved(services, use_templates=True)
    
    # Generate code
    print(f"\n{'='*70}")
    print("GENERATING TERRAFORM CODE")
    print(f"{'='*70}\n")
    
    result = rag.generate_terraform(
        requirement="Secure web application with load balancer and RDS database",
        aws_service="EC2, ALB, RDS"
    )
    
    print("📄 MAIN.TF:")
    print("-" * 70)
    print(result.get("main_tf", ""))
    
    if "variables_tf" in result:
        print("\n📄 VARIABLES.TF:")
        print("-" * 70)
        print(result.get("variables_tf", ""))
    
    if "outputs_tf" in result:
        print("\n📄 OUTPUTS.TF:")
        print("-" * 70)
        print(result.get("outputs_tf", ""))
    
    print("\n💡 EXPLANATION:")
    print("-" * 70)
    print(result.get("explanation", ""))