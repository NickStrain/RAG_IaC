import requests
import json
import time
from typing import List, Dict
import base64
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import re
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

class TerraformRegistryScraper:
    def __init__(self, github_token: str = None):
        self.session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        if github_token:
            headers['Authorization'] = f'token {github_token}'
        self.session.headers.update(headers)
        
    def get_doc_content_from_api(self, path: str) -> str:
        """Get content directly from GitHub API"""
        try:
            url = f"https://api.github.com/repos/hashicorp/terraform-provider-aws/contents/website/docs/{path}"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
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
    
    def extract_code_blocks(self, content: str) -> List[str]:
        """Extract all code blocks from markdown"""
        code_blocks = []
        lines = content.split('\n')
        in_code_block = False
        current_block = []
        
        for line in lines:
            if line.strip().startswith('```'):
                if in_code_block:
                    # End of code block
                    if current_block:
                        code_blocks.append('\n'.join(current_block))
                    current_block = []
                    in_code_block = False
                else:
                    # Start of code block
                    in_code_block = True
            elif in_code_block:
                current_block.append(line)
        
        return code_blocks
    
    def extract_sections(self, content: str) -> Dict[str, str]:
        """Extract all major sections from markdown documentation"""
        lines = content.split('\n')
        sections = {}
        current_section = 'introduction'
        current_content = []
        
        for line in lines:
            # Skip YAML front matter
            if line.strip() == '---':
                continue
            
            # Check if this is a header line
            if line.startswith('## '):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.replace('## ', '').strip().lower().replace(' ', '_')
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def parse_markdown_doc_comprehensive(self, content: str, name: str, doc_type: str) -> Dict:
        """Comprehensively parse markdown documentation including all code and sections"""
        
        # Extract all code blocks
        code_blocks = self.extract_code_blocks(content)
        
        # Extract all sections
        sections = self.extract_sections(content)
        
        # Get main description (first few paragraphs before first header)
        lines = content.split('\n')
        description_lines = []
        for line in lines:
            if line.startswith('## '):
                break
            if line.strip() and not line.startswith('#') and not line.startswith('```'):
                if not line.startswith('->') and not line.startswith('~>') and not line.strip() == '---':
                    description_lines.append(line.strip())
        
        description = ' '.join(description_lines[:10]) if description_lines else 'No description available'
        
        # Combine all code blocks
        all_code = '\n\n'.join(code_blocks)
        
        # Extract specific sections
        example_usage = sections.get('example_usage', '')
        argument_reference = sections.get('argument_reference', '') or sections.get('arguments', '')
        attribute_reference = sections.get('attribute_reference', '') or sections.get('attributes', '')
        
        return {
            'name': name,
            'type': doc_type,
            'description': description[:3000],
            'example_usage': example_usage[:5000],
            'all_code_blocks': all_code[:8000],
            'argument_reference': argument_reference[:5000],
            'attribute_reference': attribute_reference[:5000],
            'sections': sections,
            'full_text': content[:15000],  # Store more of the full text
            'code_block_count': len(code_blocks)
        }
    
    def scrape_github_docs(self, limit: int = 500, use_api: bool = True) -> List[Dict]:
        """Scrape documentation from GitHub repository"""
        print("Fetching comprehensive documentation from GitHub...")
        
        documents = []
        resources_fetched = 0
        datasources_fetched = 0
        
        try:
            # Get resources
            print("\n" + "="*70)
            print("FETCHING RESOURCES")
            print("="*70)
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
                            doc = self.parse_markdown_doc_comprehensive(content, f"aws_{name}", "resource")
                            doc['url'] = f"https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/{name}"
                            documents.append(doc)
                            resources_fetched += 1
                            if resources_fetched % 10 == 0:
                                print(f"✓ Scraped {resources_fetched}/{target_resources} resources...")
                        else:
                            print(f"✗ Failed: aws_{name}")
                        
                        time.sleep(0.3)  # Be nice to GitHub
            else:
                print(f"Failed to get resources list: {response.status_code}")
            
            # Get data sources
            print("\n" + "="*70)
            print("FETCHING DATA SOURCES")
            print("="*70)
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
                            doc = self.parse_markdown_doc_comprehensive(content, f"aws_{name}", "data_source")
                            doc['url'] = f"https://registry.terraform.io/providers/hashicorp/aws/latest/docs/data-sources/{name}"
                            documents.append(doc)
                            datasources_fetched += 1
                            if datasources_fetched % 10 == 0:
                                print(f"✓ Scraped {datasources_fetched}/{target_datasources} data sources...")
                        else:
                            print(f"✗ Failed: aws_{name}")
                        
                        time.sleep(0.3)  # Be nice to GitHub
            else:
                print(f"Failed to get data sources list: {response.status_code}")
        
        except Exception as e:
            print(f"Error scraping GitHub: {e}")
        
        print(f"\n{'='*70}")
        print(f"SCRAPING SUMMARY")
        print(f"{'='*70}")
        print(f"Total documents scraped: {len(documents)}")
        print(f"  - Resources: {resources_fetched}")
        print(f"  - Data sources: {datasources_fetched}")
        print(f"{'='*70}")
        
        return documents


class PineconeVectorDatabase:
    def __init__(self, api_key: str, index_name: str = "iac-terraform-v2", dimension: int = 384):
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
        """Add comprehensive documents to Pinecone"""
        if not documents:
            print("No documents to add!")
            return
        
        print(f"\n{'='*70}")
        print(f"ADDING DOCUMENTS TO PINECONE")
        print(f"{'='*70}")
        print(f"Total documents to add: {len(documents)}")
        
        vectors_to_upsert = []
        
        for i, doc in enumerate(documents):
            # Create comprehensive text for embedding - include all important content
            text_to_embed = f"{doc['name']} {doc['type']} {doc['description']} {doc.get('example_usage', '')} {doc.get('argument_reference', '')} {doc.get('attribute_reference', '')}"
            
            # Generate embedding
            embedding = self.model.encode(text_to_embed).tolist()
            
            # Create metadata with all extracted content (respecting Pinecone size limits)
            metadata = {
                'name': doc['name'],
                'type': doc['type'],
                'description': doc['description'][:2000],
                'url': doc['url'],
                'example_usage': doc.get('example_usage', '')[:3000],
                'all_code_blocks': doc.get('all_code_blocks', '')[:4000],
                'argument_reference': doc.get('argument_reference', '')[:2000],
                'attribute_reference': doc.get('attribute_reference', '')[:2000],
                'code_block_count': doc.get('code_block_count', 0),
                'full_text_preview': doc.get('full_text', '')[:3000]
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
        print(f"\n{'='*70}")
        print(f"INDEX STATISTICS")
        print(f"{'='*70}")
        print(f"Total vectors in index: {stats['total_vector_count']}")
        print(f"{'='*70}")
    
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
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Optional: for higher rate limits
    
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY environment variable not set")
        print("Please set it with: export PINECONE_API_KEY='your-api-key'")
        return
    
    # Scrape documentation
    print("="*70)
    print("TERRAFORM AWS DOCUMENTATION SCRAPER")
    print("Comprehensive Content Extraction")
    print("="*70)
    
    scraper = TerraformRegistryScraper(github_token=GITHUB_TOKEN)
    
    # Scrape documents (500 total: 250 resources + 250 data sources)
    # Change use_api=False to avoid rate limits, or use_api=True with GITHUB_TOKEN
    documents = scraper.scrape_github_docs(limit=500, use_api=False)
    
    if not documents:
        print("\n❌ No documents scraped. Possible issues:")
        print("   - Rate limiting from GitHub")
        print("   - Network connectivity issues")
        print("   - File path or format changes")
        return
    
    print(f"\n✓ Successfully scraped {len(documents)} documents")
    
    # Save raw documents as backup
    with open('terraform_aws_docs_comprehensive.json', 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2)
    print(f"✓ Raw documents saved to terraform_aws_docs_comprehensive.json")
    
    # Create and populate Pinecone database
    print("\n" + "="*70)
    print("CREATING PINECONE VECTOR DATABASE")
    print("="*70)
    
    vector_db = PineconeVectorDatabase(
        api_key=PINECONE_API_KEY,
        index_name="iac-terraform-v2"
    )
    
    vector_db.add_documents(documents)
    
    # Example searches
    print("\n" + "="*70)
    print("EXAMPLE SEARCHES")
    print("="*70)
    
    queries = [
        "S3 bucket with encryption example code",
        "EC2 instance with security groups terraform",
        "Lambda function example with IAM role",
        "VPC configuration with subnets"
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
            print(f"   Code blocks: {metadata.get('code_block_count', 0)}")
            if metadata.get('description'):
                desc = metadata['description'][:200]
                print(f"   Description: {desc}...")
            
            # Show a snippet of code if available
            if metadata.get('all_code_blocks'):
                code_snippet = metadata['all_code_blocks'][:300]
                print(f"   Code snippet:\n{code_snippet}...")
    
    print("\n" + "="*70)
    print("✓ SCRAPING AND INDEXING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()