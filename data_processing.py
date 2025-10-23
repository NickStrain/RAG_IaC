import requests
import json
import time
from typing import List, Dict
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import base64
from sentence_transformers import SentenceTransformer
import os

from dotenv import load_dotenv
import re
import time

load_dotenv()

class TerraformRegistryScraper:
    def __init__(self, github_token: str = None):
        self.session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        # Add GitHub token if provided to increase rate limits
        if github_token:
            headers['Authorization'] = f'token {github_token}'
        self.session.headers.update(headers)
        self.base_api_url = "https://registry.terraform.io/v2/providers/hashicorp/aws"
        
    def get_doc_content_from_api(self, path: str) -> str:
        """Get content directly from GitHub API"""
        try:
            url = f"https://api.github.com/repos/hashicorp/terraform-provider-aws/contents/website/docs/{path}"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                # Content is base64 encoded in the API response
                if 'content' in data:
                    content = base64.b64decode(data['content']).decode('utf-8')
                    return content
            elif response.status_code == 403:
                print(f"Rate limited! Waiting 60 seconds...")
                time.sleep(60)
                return None
            
            return None
        except Exception as e:
            print(f"Error fetching {path}: {e}")
            return None
    
    def get_doc_content_raw(self, path: str) -> str:
        """Get content from raw GitHub URL (fallback)"""
        try:
            url = f"https://raw.githubusercontent.com/hashicorp/terraform-provider-aws/main/website/docs/{path}"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                return response.text
            
            return None
        except Exception as e:
            print(f"Error fetching raw {path}: {e}")
            return None
    
    def parse_markdown_doc(self, content: str, name: str, doc_type: str) -> Dict:
        """Parse markdown documentation"""
        lines = content.split('\n')
        
        description = []
        example = []
        arguments = []
        attributes = []
        in_example = False
        in_description = True
        in_arguments = False
        in_attributes = False
        
        for line in lines:
            # Skip front matter
            if line.strip() == '---':
                continue
            
            # Check for sections
            if '## Argument Reference' in line or '## Arguments' in line:
                in_arguments = True
                in_attributes = False
                in_description = False
            elif '## Attribute Reference' in line or '## Attributes' in line:
                in_attributes = True
                in_arguments = False
                in_description = False
            elif line.startswith('## '):
                in_arguments = False
                in_attributes = False
                
            # Extract description (first few non-empty, non-header paragraphs)
            if in_description and line.strip() and not line.startswith('#') and not line.startswith('```'):
                if not line.startswith('->') and not line.startswith('~>'):
                    description.append(line.strip())
                    if len(description) >= 5:
                        in_description = False
            
            # Extract arguments
            if in_arguments and line.strip().startswith('*'):
                arguments.append(line.strip())
            
            # Extract attributes
            if in_attributes and line.strip().startswith('*'):
                attributes.append(line.strip())
            
            # Extract example code
            if '```' in line:
                if 'terraform' in line.lower() or 'hcl' in line.lower():
                    in_example = not in_example
                elif in_example:
                    in_example = False
            elif in_example:
                example.append(line)
                if len(example) >= 100:
                    in_example = False
        
        return {
            'name': name,
            'type': doc_type,
            'description': ' '.join(description)[:2000] if description else 'No description available',
            'example': '\n'.join(example[:100]) if example else '',
            'arguments': ' '.join(arguments[:50]) if arguments else '',
            'attributes': ' '.join(attributes[:50]) if attributes else '',
            'full_text': content[:10000]
        }
    
    def scrape_github_docs(self, limit: int = 500, use_api: bool = True) -> List[Dict]:
        """Scrape documentation from GitHub repository"""
        print("Fetching documentation list from GitHub...")
        
        documents = []
        resources_fetched = 0
        datasources_fetched = 0
        
        try:
            # Get resources
            print("\nFetching resources list...")
            resources_url = "https://api.github.com/repos/hashicorp/terraform-provider-aws/contents/website/docs/r"
            response = self.session.get(resources_url, timeout=15)
            
            if response.status_code == 200:
                resources = response.json()
                target_resources = min(limit // 2, len(resources))
                print(f"Found {len(resources)} resources. Scraping up to {target_resources}...")
                
                for i, resource in enumerate(resources[:target_resources]):
                    if resource['name'].endswith('.markdown') or resource['name'].endswith('.md'):
                        name = resource['name'].replace('.html.markdown', '').replace('.markdown', '').replace('.md', '')
                        
                        # Try API first, then fallback to raw
                        if use_api:
                            content = self.get_doc_content_from_api(f"r/{resource['name']}")
                        else:
                            content = None
                            
                        if not content:
                            content = self.get_doc_content_raw(f"r/{resource['name']}")
                        
                        if content:
                            doc = self.parse_markdown_doc(content, f"aws_{name}", "resource")
                            doc['url'] = f"https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/{name}"
                            documents.append(doc)
                            resources_fetched += 1
                            if resources_fetched % 10 == 0:
                                print(f"✓ Scraped {resources_fetched} resources...")
                        else:
                            print(f"✗ Failed: aws_{name}")
                        
                        time.sleep(0.3)  # Be nice to GitHub
            else:
                print(f"Failed to get resources list: {response.status_code}")
            
            # Get data sources
            print(f"\nFetching data sources list...")
            data_sources_url = "https://api.github.com/repos/hashicorp/terraform-provider-aws/contents/website/docs/d"
            response = self.session.get(data_sources_url, timeout=15)
            
            if response.status_code == 200:
                data_sources = response.json()
                target_datasources = min(limit // 2, len(data_sources))
                print(f"Found {len(data_sources)} data sources. Scraping up to {target_datasources}...")
                
                for i, ds in enumerate(data_sources[:target_datasources]):
                    if ds['name'].endswith('.markdown') or ds['name'].endswith('.md'):
                        name = ds['name'].replace('.html.markdown', '').replace('.markdown', '').replace('.md', '')
                        
                        # Try API first, then fallback to raw
                        if use_api:
                            content = self.get_doc_content_from_api(f"d/{ds['name']}")
                        else:
                            content = None
                            
                        if not content:
                            content = self.get_doc_content_raw(f"d/{ds['name']}")
                        
                        if content:
                            doc = self.parse_markdown_doc(content, f"aws_{name}", "data_source")
                            doc['url'] = f"https://registry.terraform.io/providers/hashicorp/aws/latest/docs/data-sources/{name}"
                            documents.append(doc)
                            datasources_fetched += 1
                            if datasources_fetched % 10 == 0:
                                print(f"✓ Scraped {datasources_fetched} data sources...")
                        else:
                            print(f"✗ Failed: aws_{name}")
                        
                        time.sleep(0.3)  # Be nice to GitHub
            else:
                print(f"Failed to get data sources list: {response.status_code}")
        
        except Exception as e:
            print(f"Error scraping GitHub: {e}")
        
        print(f"\n{'='*70}")
        print(f"Summary: Scraped {len(documents)} total documents")
        print(f"  - Resources: {resources_fetched}")
        print(f"  - Data sources: {datasources_fetched}")
        print(f"{'='*70}")
        
        return documents


class PineconeVectorDatabase:
    def __init__(self, api_key: str, index_name: str = "terraform-docs", dimension: int = 384):
        """Initialize Pinecone vector database"""
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        
        # Initialize sentence transformer for embeddings
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
        
        # Create index if it doesn't exist
        self._create_index_if_not_exists()
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to Pinecone index: {self.index_name}")
    
    def _create_index_if_not_exists(self):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            # Wait for index to be ready
            time.sleep(10)
            print(f"Index {self.index_name} created successfully")
        else:
            print(f"Index {self.index_name} already exists")
    
    def add_documents(self, documents: List[Dict], batch_size: int = 100):
        """Add documents to Pinecone"""
        if not documents:
            print("No documents to add!")
            return
        
        print(f"Adding {len(documents)} documents to Pinecone...")
        
        vectors_to_upsert = []
        
        for i, doc in enumerate(documents):
            # Create text for embedding
            text_to_embed = f"{doc['name']} {doc['type']} {doc['description']} {doc.get('arguments', '')} {doc.get('attributes', '')}"
            
            # Generate embedding
            embedding = self.model.encode(text_to_embed).tolist()
            
            # Create metadata (Pinecone has size limits, so we truncate)
            metadata = {
                'name': doc['name'],
                'type': doc['type'],
                'description': doc['description'][:1000],
                'url': doc['url'],
                'example': doc.get('example', '')[:1000],
                'arguments': doc.get('arguments', '')[:500],
                'attributes': doc.get('attributes', '')[:500]
            }
            
            # Add to batch
            vectors_to_upsert.append({
                'id': f"{doc['type']}_{doc['name']}_{i}",
                'values': embedding,
                'metadata': metadata
            })
            
            # Upsert in batches
            if len(vectors_to_upsert) >= batch_size:
                self.index.upsert(vectors=vectors_to_upsert)
                print(f"✓ Uploaded {i+1}/{len(documents)} documents")
                vectors_to_upsert = []
        
        # Upsert remaining vectors
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
        
        print(f"✓ All {len(documents)} documents added to Pinecone")
        
        # Get index stats
        stats = self.index.describe_index_stats()
        print(f"Index now contains {stats['total_vector_count']} vectors")
    
    def search(self, query: str, top_k: int = 5, filter_dict: Dict = None) -> List[Dict]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self.model.encode(query).tolist()
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Format results
        formatted_results = []
        for match in results['matches']:
            formatted_results.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': match['metadata']
            })
        
        return formatted_results
    
    def delete_all(self):
        """Delete all vectors from the index"""
        self.index.delete(delete_all=True)
        print("All vectors deleted from index")


def main():
    # Configuration
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')  # Set this as environment variable
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Optional: for higher rate limits
    
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY environment variable not set")
        print("Please set it with: export PINECONE_API_KEY='your-api-key'")
        return
    
    # Scrape documentation
    print("="*70)
    print("STEP 1: SCRAPING TERRAFORM DOCUMENTATION")
    print("="*70)
    
    scraper = TerraformRegistryScraper(github_token=GITHUB_TOKEN)
    
    # Scrape more documents (500 total: 250 resources + 250 data sources)
    documents = scraper.scrape_github_docs(limit=500, use_api=False)
    
    if not documents:
        print("\n❌ No documents scraped. Possible issues:")
        print("   - Rate limiting from GitHub")
        print("   - Network connectivity issues")
        print("   - File path or format changes")
        return
    
    print(f"\n✓ Successfully scraped {len(documents)} documents")
    
    # Save raw documents as backup
    with open('terraform_aws_docs.json', 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2)
    print("✓ Raw documents saved to terraform_aws_docs.json")
    
    # Create and populate Pinecone database
    print("\n" + "="*70)
    print("STEP 2: CREATING PINECONE VECTOR DATABASE")
    print("="*70)
    
    vector_db = PineconeVectorDatabase(
        api_key=PINECONE_API_KEY,
        index_name="terraform-aws-docs"
    )
    
    vector_db.add_documents(documents)
    
    # Example searches
    print("\n" + "="*70)
    print("STEP 3: EXAMPLE SEARCHES")
    print("="*70)
    
    queries = [
        "EC2 instance with security groups",
        "S3 bucket with encryption",
        "VPC with subnets and routing",
        "Lambda function with IAM role",
        "RDS database with multi-az",
        "CloudWatch alarms and monitoring",
        "ECS container service",
        "API Gateway REST API"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 70)
        results = vector_db.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            score = result['score']
            print(f"\n{i}. {metadata['name']} ({metadata['type']}) - Score: {score:.3f}")
            print(f"   URL: {metadata['url']}")
            if metadata.get('description'):
                desc = metadata['description'][:150]
                print(f"   Description: {desc}...")
    
    # Search by type filter
    print("\n" + "="*70)
    print("FILTERED SEARCH EXAMPLE (Resources only)")
    print("="*70)
    
    print(f"\n🔍 Query: 'database' (resources only)")
    print("-" * 70)
    results = vector_db.search(
        "database",
        top_k=5,
        filter_dict={"type": {"$eq": "resource"}}
    )
    
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        score = result['score']
        print(f"{i}. {metadata['name']} - Score: {score:.3f}")
        print(f"   {metadata['url']}")


if __name__ == "__main__":
    main()