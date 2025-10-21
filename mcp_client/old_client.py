import sys
import json
import asyncio

from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition



class MCPClient:
    def __init__(self, model_name="qwen3:4b"):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = ChatOllama(model=model_name)  # Ollama model, e.g., "qwen3:8b"
        self.available_tools = None

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, 
            args=[server_script_path], 
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        response = await self.session.list_tools()
        self.available_tools = response.tools
        
        print("\nConnected to server with tools:", [tool.name for tool in self.available_tools])


    async def process_query(self, query: str) -> str:
        """
        Process a query using LLM and available tools
        """
        
        available_tools = await load_mcp_tools(self.session)

        agent = create_react_agent(
            model = self.llm, 
            tools = available_tools
        )
        
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user", 
                        "content": query
                    }
                ]
            }
        )
        
        response = result.get("output", str(result))
        
        response_data = json.dumps(response)
        print(response_data)


        return response

    async def chat_loop(self):
        """
        Run an interactive chat loop
        """
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        while True:
            try:
                
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                
                response = await self.process_query(query)
                print("\n" + response)
            
            except Exception as e:
                print(f"\nError: {str(e)}")
                raise e

    async def cleanup(self):
        """
        Clean up resources
        """
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
    
    client = MCPClient(model_name="qwen3:4b")
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
