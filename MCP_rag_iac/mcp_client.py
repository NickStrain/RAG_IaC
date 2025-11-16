import asyncio
import sys
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables from .env

class TerraformMCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Configure Gemini with new SDK
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment")
        
        self.client = genai.Client(api_key=api_key)
        # Use stable model instead of experimental to avoid rate limits
        self.model_name = os.getenv('GEMINI_MODEL', "gemini-2.5-flash")
        
    async def connect_to_docker_server(self, image_name: str = "hashicorp/terraform-mcp-server:0.3.0", env_vars: dict = None):
        """Connect to a Terraform MCP server using Docker run
        
        Args:
            image_name: Docker image name (e.g., 'hashicorp/terraform-mcp-server:0.3.0')
            env_vars: Environment variables to pass to the container
        """
        print(f"\nStarting Docker container with image: {image_name}...")
        
        # Prepare environment variables
        env = os.environ.copy()
        
        # Build docker run command - EXACT same order as Claude Desktop config
        docker_args = ["run", "-i", "--rm"]
        
        # Add environment variables in exact same format
        if env_vars:
            for key, value in env_vars.items():
                docker_args.extend(["-e", f"{key}={value}"])
        
        docker_args.append(image_name)
        
        server_params = StdioServerParameters(
            command="docker",
            args=docker_args,
            env=None  # Don't pass env here, put everything in args
        )
        
        try:
            # Print command with masked token
            safe_args = []
            for arg in docker_args:
                if 'TFE_TOKEN=' in arg:
                    safe_args.append('TFE_TOKEN=***')
                else:
                    safe_args.append(arg)
            print(f"Command: docker {' '.join(safe_args)}")
            
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            
            print("Initializing session...")
            await self.session.initialize()
            
            # List available tools
            print("Fetching available tools...")
            response = await self.session.list_tools()
            tools = response.tools
            print(f"\n✓ Connected to Terraform MCP server with {len(tools)} tools:")
            for i, tool in enumerate(tools[:10], 1):  # Show first 10 tools
                print(f"  {i}. {tool.name}")
            if len(tools) > 10:
                print(f"  ... and {len(tools) - 10} more tools")
                
        except Exception as e:
            print(f"\n✗ Failed to start Docker container")
            print(f"Error: {str(e)}")
            print(f"\nTroubleshooting tips:")
            print(f"1. Verify Docker is installed and running: docker --version")
            print(f"2. Check if image exists: docker images | grep terraform-mcp-server")
            print(f"3. Test manually with the exact command from Claude Desktop config")
            print(f"4. Verify TFE_TOKEN and TFE_ADDRESS are correct in .env file")
            raise
    
    async def connect_to_server(self, server_script_path: str):
        """Connect to a local MCP server script
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        # Validate file exists
        if not os.path.exists(server_script_path):
            raise FileNotFoundError(f"Server script not found: {server_script_path}")
        
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        
        command = "python" if is_python else "node"
        
        print(f"\nStarting MCP server...")
        print(f"Command: {command} {server_script_path}")
        
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=os.environ.copy()
        )
        
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            
            print("Initializing session...")
            await self.session.initialize()
            
            # List available tools
            print("Fetching available tools...")
            response = await self.session.list_tools()
            tools = response.tools
            print(f"\n✓ Connected to Terraform MCP server with {len(tools)} tools")
                
        except Exception as e:
            print(f"\n✗ Failed to connect to MCP server")
            print(f"Error: {str(e)}")
            raise
    
    def convert_mcp_tools_to_gemini(self, mcp_tools):
        """Convert MCP tool format to Gemini function calling format"""
        gemini_tools = []
        
        for tool in mcp_tools:
            # Create function declaration compatible with new SDK
            function_declaration = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema
            )
            gemini_tools.append(function_declaration)
        
        # Wrap in Tool object
        return [types.Tool(function_declarations=gemini_tools)]
    
    async def process_query(self, query: str) -> str:
        """Process a query using Gemini and available Terraform tools"""
        
        # Get available tools from MCP server
        response = await self.session.list_tools()
        mcp_tools = response.tools
        gemini_tools = self.convert_mcp_tools_to_gemini(mcp_tools)
        
        # Build conversation history
        messages = [types.Content(
            role="user",
            parts=[types.Part(text=query)]
        )]
        
        final_text = []
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            try:
                # Generate content with tools
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=messages,
                    config=types.GenerateContentConfig(
                        tools=gemini_tools,
                        temperature=0.7
                    )
                )
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    return (
                        "❌ Gemini API quota exceeded. Please try:\n"
                        "1. Wait a minute and try again\n"
                        "2. Use a different model: Set GEMINI_MODEL=gemini-1.5-flash in .env\n"
                        "3. Check your quota at: https://ai.dev/usage\n"
                        "4. Upgrade your Gemini API plan for higher limits\n\n"
                        f"Error: {error_msg}"
                    )
                else:
                    raise
            
            # Add assistant response to history
            assistant_parts = []
            
            # Check for text response
            has_function_calls = False
            
            for part in response.candidates[0].content.parts:
                assistant_parts.append(part)
                
                if hasattr(part, 'text') and part.text:
                    final_text.append(part.text)
                
                if hasattr(part, 'function_call') and part.function_call:
                    has_function_calls = True
            
            # Add assistant message to history
            messages.append(types.Content(
                role="model",
                parts=assistant_parts
            ))
            
            # If no function calls, we're done
            if not has_function_calls:
                break
            
            # Process function calls
            function_responses = []
            
            for part in assistant_parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_call = part.function_call
                    tool_name = function_call.name
                    tool_args = dict(function_call.args)
                    
                    print(f"\n[Calling tool: {tool_name}]")
                    print(f"[Arguments: {json.dumps(tool_args, indent=2)}]")
                    
                    # Execute tool call via MCP
                    try:
                        result = await self.session.call_tool(tool_name, tool_args)
                        
                        # Format result - handle MCP content types
                        result_content = result.content
                        
                        # Handle list of content items (MCP standard format)
                        if isinstance(result_content, list):
                            text_parts = []
                            for item in result_content:
                                if hasattr(item, 'text'):
                                    text_parts.append(item.text)
                                elif hasattr(item, 'type') and item.type == 'text':
                                    text_parts.append(str(item))
                                elif isinstance(item, dict) and 'text' in item:
                                    text_parts.append(item['text'])
                                else:
                                    text_parts.append(str(item))
                            result_content = '\n'.join(text_parts)
                        elif hasattr(result_content, 'text'):
                            result_content = result_content.text
                        elif not isinstance(result_content, str):
                            result_content = str(result_content)
                        
                        function_responses.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=tool_name,
                                    response={"result": result_content}
                                )
                            )
                        )
                        
                        print(f"[Tool result received]")
                        
                    except Exception as e:
                        error_msg = f"Error executing tool: {str(e)}"
                        print(f"[{error_msg}]")
                        function_responses.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=tool_name,
                                    response={"error": error_msg}
                                )
                            )
                        )
            
            # Add function responses to conversation
            if function_responses:
                messages.append(types.Content(
                    role="user",
                    parts=function_responses
                ))
            else:
                break
        
        return "\n".join(final_text) if final_text else "No response generated."
    
    async def chat_loop(self):
        """Run an interactive chat loop for Terraform operations"""
        print("\n" + "="*60)
        print("Terraform MCP Client with Gemini Started!")
        print("="*60)
        print("\nYou can ask me to help with Terraform operations like:")
        print("  - Search for providers and modules")
        print("  - Get latest versions")
        print("  - Manage workspaces")
        print("  - Create and manage runs")
        print("  - Handle variables and variable sets")
        print("  - And more!")
        print("\nType your queries or 'quit' to exit.")
        print("="*60)
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not query:
                    continue
                
                print("\n[Processing...]")
                response = await self.process_query(query)
                print("\n" + "─"*60)
                print(response)
                print("─"*60)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    print(f"\n❌ API quota exceeded. Please wait a minute and try again.")
                    print(f"Tip: Add 'GEMINI_MODEL=gemini-1.5-flash' to your .env for stable quota")
                else:
                    print(f"\nError: {error_str}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


    async def connect_to_npx_server(self, package_name: str, args: list = None):
        """Connect to an MCP server via npx
        
        Args:
            package_name: NPM package name (e.g., '@modelcontextprotocol/server-terraform')
            args: Additional arguments to pass to the server
        """
        print(f"\nStarting MCP server via npx...")
        print(f"Package: {package_name}")
        
        # Prepare environment variables
        env = os.environ.copy()
        
        # Build command arguments
        cmd_args = [package_name]
        if args:
            cmd_args.extend(args)
        
        server_params = StdioServerParameters(
            command="npx",
            args=cmd_args,
            env=env
        )
        
        try:
            print(f"Command: npx {' '.join(cmd_args)}")
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            
            print("Initializing session...")
            await self.session.initialize()
            
            # List available tools
            print("Fetching available tools...")
            response = await self.session.list_tools()
            tools = response.tools
            print(f"\n✓ Connected to Terraform MCP server with {len(tools)} tools")
            
            # Show first few tools
            for i, tool in enumerate(tools[:5], 1):
                print(f"  {i}. {tool.name}")
            if len(tools) > 5:
                print(f"  ... and {len(tools) - 5} more tools")
                
        except Exception as e:
            print(f"\n✗ Failed to connect via npx")
            print(f"Error: {str(e)}")
            print(f"\nTroubleshooting tips:")
            print(f"1. Ensure Node.js and npx are installed: npx --version")
            print(f"2. Check if package exists: npm view {package_name}")
            print(f"3. Try running directly: npx {package_name}")
            raise


async def main():
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables.")
        print("Please add it to your .env file:")
        print("  GOOGLE_API_KEY=your_api_key_here")
        sys.exit(1)
    
    client = TerraformMCPClient()
    
    # Determine connection mode
    if len(sys.argv) >= 2:
        if sys.argv[1] == "--docker":
            image_name = sys.argv[2] if len(sys.argv) > 2 else "hashicorp/terraform-mcp-server:0.3.0"
            
            # Get environment variables for Terraform from .env
            env_vars = {}
            
            # Required variables
            tfe_token = os.getenv('TFE_TOKEN')
            if not tfe_token:
                print("Error: TFE_TOKEN not found in environment variables.")
                print("Please add it to your .env file:")
                print("  TFE_TOKEN=your_terraform_cloud_token")
                sys.exit(1)
            
            env_vars['TFE_TOKEN'] = tfe_token
            
            # Optional variables with defaults
            tfe_address = os.getenv('TFE_ADDRESS', 'https://app.terraform.io/app/shuga/workspaces')
            env_vars['TFE_ADDRESS'] = tfe_address
            
            try:
                await client.connect_to_docker_server(image_name, env_vars)
                await client.chat_loop()
            finally:
                await client.cleanup()
        elif sys.argv[1] == "--npx":
            package_name = sys.argv[2] if len(sys.argv) > 2 else "@modelcontextprotocol/server-terraform"
            try:
                await client.connect_to_npx_server(package_name)
                await client.chat_loop()
            finally:
                await client.cleanup()
        else:
            # Assume it's a file path
            server_path = sys.argv[1]
            try:
                await client.connect_to_server(server_path)
                await client.chat_loop()
            finally:
                await client.cleanup()
    else:
        print("Usage:")
        print("  For Docker: python mcp_client.py --docker [image_name]")
        print("  For NPX:    python mcp_client.py --npx [package_name]")
        print("  For local:  python mcp_client.py <path_to_server_script>")
        print("\nExamples:")
        print("  python mcp_client.py --docker")
        print("  python mcp_client.py --docker hashicorp/terraform-mcp-server:0.3.0")
        print("  python mcp_client.py --npx")
        print("  python mcp_client.py ./terraform-server.js")
        print("\nRequired environment variables in .env file:")
        print("  GOOGLE_API_KEY=your_gemini_key")
        print("  TFE_TOKEN=your_terraform_cloud_token")
        print("\nOptional environment variables:")
        print("  TFE_ADDRESS=https://app.terraform.io/app/your-org/workspaces")
        print("  GEMINI_MODEL=gemini-1.5-flash  # Use stable model to avoid rate limits")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())