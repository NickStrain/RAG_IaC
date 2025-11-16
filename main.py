import asyncio
import sys
import os
from typing import Optional, Dict
from dotenv import load_dotenv
from google import genai

# Import your existing modules
from mcp_client import TerraformMCPClient
from main1 import RAGSystem, PineconeIndex

load_dotenv()


class QueryRouter:
    """Intelligent query router that determines intent"""
    
    def __init__(self, gemini_client: genai.Client, model_name: str):
        self.gemini_client = gemini_client
        self.model_name = model_name
    
    def classify_query(self, query: str) -> Dict:
        """Classify user query into RAG or MCP category"""
        
        prompt = f"""You are an intelligent query classifier for a Terraform system. Analyze this user query and determine the intent.

USER QUERY: "{query}"

CLASSIFICATION RULES:

**RAG System (Code Generation)** - Use when user wants to:
- Generate new Terraform code
- Create infrastructure code
- Write Terraform configuration
- Build/Deploy new resources
- Design infrastructure
- Keywords: "create", "generate", "write", "build", "deploy", "design", "code for", "terraform for"

**MCP System (Terraform Operations)** - Use when user wants to:
- List/search existing resources (workspaces, modules, providers, runs, organizations)
- Get details about existing resources
- Manage existing infrastructure
- Execute Terraform operations (init, plan, apply)
- Query Terraform state
- Manage variables or variable sets
- View workspace information
- Check run status
- Keywords: "list", "show", "get", "search", "find", "status", "details", "workspace", "run", "organization"

Respond ONLY in JSON format:
{{
    "intent": "rag" or "mcp",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "suggested_action": "what the system should do"
}}
"""
        
        try:
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            import re
            import json
            
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                classification = json.loads(json_match.group())
                return classification
        except Exception as e:
            print(f"Classification error: {e}")
        
        # Fallback: Simple keyword-based classification
        query_lower = query.lower()
        
        # MCP keywords
        mcp_keywords = ['list', 'show', 'get', 'search', 'find', 'workspace', 
                       'organization', 'run', 'status', 'details', 'view']
        
        # RAG keywords  
        rag_keywords = ['create', 'generate', 'write', 'build', 'deploy', 
                       'design', 'code', 'terraform for', 'infrastructure for']
        
        mcp_score = sum(1 for kw in mcp_keywords if kw in query_lower)
        rag_score = sum(1 for kw in rag_keywords if kw in query_lower)
        
        if mcp_score > rag_score:
            intent = "mcp"
            confidence = min(0.7, 0.5 + (mcp_score * 0.1))
        else:
            intent = "rag"
            confidence = min(0.7, 0.5 + (rag_score * 0.1))
        
        return {
            "intent": intent,
            "confidence": confidence,
            "reasoning": f"Keyword analysis: MCP={mcp_score}, RAG={rag_score}",
            "suggested_action": "Route to appropriate system"
        }


class UnifiedTerraformSystem:
    """Unified system that routes queries to RAG or MCP"""
    
    def __init__(self):
        self.gemini_client = None
        self.model_name = os.getenv('GEMINI_MODEL', "gemini-2.5-flash")
        self.mcp_client = None
        self.rag_system = None
        self.router = None
        
    async def initialize(self):
        """Initialize all components"""
        print("🔧 Initializing Unified Terraform System...\n")
        
        # Initialize Gemini
        try:
            api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found")
            
            self.gemini_client = genai.Client(api_key=api_key)
            print("✓ Gemini client initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            raise
        
        # Initialize Query Router
        self.router = QueryRouter(self.gemini_client, self.model_name)
        print("✓ Query router initialized")
        
        # Initialize MCP Client
        try:
            self.mcp_client = TerraformMCPClient()
            
            # Connect to Terraform MCP server
            image_name = "hashicorp/terraform-mcp-server:0.3.0"
            env_vars = {}
            
            tfe_token = os.getenv('TFE_TOKEN')
            if tfe_token:
                env_vars['TFE_TOKEN'] = tfe_token
                tfe_address = os.getenv('TFE_ADDRESS', 'https://app.terraform.io')
                env_vars['TFE_ADDRESS'] = tfe_address
                
                await self.mcp_client.connect_to_docker_server(image_name, env_vars)
                print("✓ MCP client connected to Terraform server")
            else:
                print("⚠ TFE_TOKEN not found - MCP features will be limited")
                self.mcp_client = None
        except Exception as e:
            print(f"⚠ MCP client initialization failed: {e}")
            print("  RAG system will still work for code generation")
            self.mcp_client = None
        
        # Initialize RAG System
        try:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
            
            if pinecone_api_key and pinecone_env:
                vector_store = PineconeIndex(
                    pinecone_api_key,
                    pinecone_env,
                    index_name="terraform-aws-docs"
                )
                
                self.rag_system = RAGSystem(
                    pinecone_index=vector_store,
                    gemini_client=self.gemini_client,
                    model_name=self.model_name
                )
                print("✓ RAG system initialized with vector store")
            else:
                print("⚠ Pinecone credentials not found - RAG features disabled")
                self.rag_system = None
        except Exception as e:
            print(f"⚠ RAG system initialization failed: {e}")
            print("  MCP operations will still work")
            self.rag_system = None
        
        print("\n" + "="*70)
        print("✅ SYSTEM READY")
        print("="*70)
        print(f"  RAG System: {'✓ Available' if self.rag_system else '✗ Unavailable'}")
        print(f"  MCP Client: {'✓ Available' if self.mcp_client else '✗ Unavailable'}")
        print("="*70 + "\n")
    
    async def process_query(self, query: str):
        """Process user query with intelligent routing"""
        
        if not query.strip():
            print("❌ Query cannot be empty")
            return
        
        print("\n" + "="*70)
        print("🔍 ANALYZING QUERY")
        print("="*70)
        print(f"Query: {query}\n")
        
        # Classify query
        classification = self.router.classify_query(query)
        
        intent = classification.get('intent', 'rag')
        confidence = classification.get('confidence', 0.5)
        reasoning = classification.get('reasoning', 'Unknown')
        
        print(f"Intent: {intent.upper()}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Reasoning: {reasoning}\n")
        
        # Route to appropriate system
        if intent == "mcp":
            if self.mcp_client:
                print("="*70)
                print("🔧 ROUTING TO: TERRAFORM MCP CLIENT")
                print("="*70)
                await self.handle_mcp_query(query)
            else:
                print("❌ MCP client not available. Please check TFE_TOKEN configuration.")
        
        elif intent == "rag":
            if self.rag_system:
                print("="*70)
                print("🤖 ROUTING TO: RAG CODE GENERATION SYSTEM")
                print("="*70)
                self.handle_rag_query(query)
            else:
                print("❌ RAG system not available. Please check Pinecone configuration.")
        
        else:
            print(f"❌ Unknown intent: {intent}")
    
    async def handle_mcp_query(self, query: str):
        """Handle MCP-related queries"""
        try:
            response = await self.mcp_client.process_query(query)
            print("\n" + "─"*70)
            print("RESULT:")
            print("─"*70)
            print(response)
            print("─"*70)
        except Exception as e:
            print(f"\n❌ MCP query failed: {e}")
    
    def handle_rag_query(self, query: str):
        """Handle RAG code generation queries"""
        try:
            result = self.rag_system.generate_terraform_code(query)
            
            # Additional summary
            print("\n" + "="*70)
            print("📋 GENERATION SUMMARY")
            print("="*70)
            print(f"  Variables provided: {len(result['variables'])}")
            print(f"  Variables used: {len(result['used_variables'])}")
            print(f"  Variables missing: {len(result['unused_variables'])}")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ RAG generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def interactive_mode(self):
        """Run in interactive mode"""
        print("\n" + "="*70)
        print("🏗️  UNIFIED TERRAFORM SYSTEM - INTERACTIVE MODE")
        print("="*70)
        print("\n💡 This system can:")
        print("  1. Generate new Terraform code (RAG)")
        print("     Example: 'Create an S3 bucket named data-prod in us-west-2'")
        print("\n  2. Query existing Terraform infrastructure (MCP)")
        print("     Example: 'List all Terraform workspaces'")
        print("     Example: 'Show details of workspace production'")
        print("\n  3. Auto-detect your intent and route accordingly")
        print("\n💬 Just type your request naturally!")
        print("   Type 'quit' to exit\n")
        print("="*70)
        
        while True:
            try:
                query = input("\n💬 Your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                await self.process_query(query)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.mcp_client:
            await self.mcp_client.cleanup()


async def main():
    """Main entry point"""
    
    # Check for required environment variables
    required_env_vars = {
        'GEMINI_API_KEY': 'Gemini API key',
        'TFE_TOKEN': 'Terraform Cloud token (for MCP operations)',
        'PINECONE_API_KEY': 'Pinecone API key (for RAG generation)',
        'PINECONE_ENVIRONMENT': 'Pinecone environment (for RAG generation)'
    }
    
    missing_vars = []
    for var, description in required_env_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  {var}: {description}")
    
    if 'GEMINI_API_KEY' not in [v.split(':')[0].strip() for v in missing_vars]:
        # At least Gemini is available
        if missing_vars:
            print("⚠️  Some features will be limited due to missing environment variables:")
            for var in missing_vars:
                print(var)
            print("\nYou can still use available features.\n")
    else:
        print("❌ GOOGLE_API_KEY is required. Please add it to your .env file:")
        print("  GOOGLE_API_KEY=your_gemini_api_key")
        sys.exit(1)
    
    system = UnifiedTerraformSystem()
    
    try:
        await system.initialize()
        
        # Check if running with a query argument
        if len(sys.argv) > 1:
            # Single query mode
            query = ' '.join(sys.argv[1:])
            await system.process_query(query)
        else:
            # Interactive mode
            await system.interactive_mode()
            
    except Exception as e:
        print(f"\n System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await system.cleanup()


if __name__ == "__main__":
    asyncio.run(main())