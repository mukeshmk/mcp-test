import asyncio
import argparse
from typing import Annotated, List, Optional, TypedDict, Tuple, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    """State class for the LangGraph agent.
    
    This TypedDict defines the state structure used throughout the agent's execution.
    It contains a list of messages that accumulates conversation history.
    
    Attributes:
        messages: A list of BaseMessage objects representing the conversation history.
                 The add_messages annotation ensures messages are properly accumulated
                 when the state is updated.
    """
    messages: Annotated[List[BaseMessage], add_messages]


class MCPClient:
    """MCP (Model Context Protocol) Client for interacting with AI models and tools.
    
    This client provides a unified interface for connecting to MCP servers, loading
    available tools, and running conversational agents with LangGraph. It supports
    both HTTP and stdio-based MCP server connections.
    
    Attributes:
        exit_stack: AsyncExitStack for managing async context managers.
        model_provider: The provider for the language model (e.g., "ollama").
        model_name: The specific model name to use.
        llm: The initialized language model instance.
        session: Optional MCP client session for server communication.
        available_tools: Optional list of loaded MCP tools.
        memory: InMemorySaver for persisting conversation state.
        graph: Cached LangGraph agent graph for conversation handling.
    """
    
    def __init__(self, model_provider: str = "ollama", model_name: str = "qwen3:4b"):
        """Initialize the MCP client.
        
        Args:
            model_provider: The provider for the language model (default: "ollama").
            model_name: The specific model name to use (default: "qwen3:4b").
        """
        self.exit_stack = AsyncExitStack()
        self.model_provider, self.model_name = model_provider, model_name
        self.llm = init_chat_model(model=self.model_name, model_provider=self.model_provider)
        self.session: Optional[ClientSession] = None
        self.available_tools: Optional[List[Any]] = None
        self.memory = InMemorySaver()
        self.graph = None  # Cached agent graph

    def extract_ai_thought(self, text: str) -> Tuple[str, str]:
        """Extract AI thought process from text containing <think> tags.
        
        This method parses text that may contain AI reasoning wrapped in <think> tags
        and separates the thought process from the actual response content.
        
        Args:
            text: The input text that may contain <think>...</think> tags.
            
        Returns:
            A tuple containing (thought_content, response_content). If no think tags
            are found or they are malformed, returns ("", original_text).
        """
        start_tag, end_tag = "<think>", "</think>"
        start_index, end_index = text.find(start_tag), text.find(end_tag)
        if start_index == -1 or end_index == -1 or end_index < start_index:
            return "", text
        between_think = text[start_index + len(start_tag):end_index].strip()
        after_think = text[end_index + len(end_tag):].strip()
        return between_think, after_think

    async def connect(self, url: Optional[str] = None, server_script_path: Optional[str] = None):
        """Connect to an MCP server and initialize the agent.
        
        This method establishes a connection to an MCP server either via HTTP or stdio,
        initializes the session, loads available tools, and builds the agent graph.
        
        Args:
            url: Optional URL for HTTP-based MCP server connection.
            server_script_path: Optional path to a Python or Node.js MCP server script
                               for stdio-based connection.
                               
        Raises:
            ValueError: If neither url nor server_script_path is provided.
            Exception: If connection or initialization fails.
        """
        try:
            if url:
                read_stream, write_stream, _ = await self.exit_stack.enter_async_context(streamablehttp_client(url))
                self.session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            elif server_script_path:
                command = "python" if server_script_path.endswith('.py') else "node"
                params = StdioServerParameters(command=command, args=[server_script_path], env=None)
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
                self.session = await self.exit_stack.enter_async_context(ClientSession(*stdio_transport))
            else:
                raise ValueError("Must provide either url or server_script_path.")

            await self.session.initialize()
            self.available_tools = await load_mcp_tools(self.session)
            tool_names = [getattr(tool, 'name', str(tool)) for tool in self.available_tools]
            print(f"\nConnected. Available tools: {tool_names}")
            await self.build_agent_graph()  # Build and cache the graph now

        except Exception as e:
            print(f"Failed to connect: {e}")
            raise

    async def call_agent(self, state: State) -> State:
        """Call the language model with the current conversation state.
        
        This method invokes the language model with the available tools bound,
        processes the response, and returns the updated state with the AI's message.
        
        Args:
            state: The current conversation state containing message history.
            
        Returns:
            Updated state containing the AI's response message.
        """
        response = await self.llm.bind_tools(self.available_tools).ainvoke(input=state["messages"])
        ai_msg = response if isinstance(response, BaseMessage) else AIMessage(content=str(response))
        return {"messages": [ai_msg]}

    async def build_agent_graph(self):
        """Build and compile the LangGraph agent graph.
        
        This method constructs a state graph that handles the conversation flow:
        - Starts with calling the agent
        - Conditionally routes to tools if tool calls are needed
        - Returns to the agent after tool execution
        - Ends the conversation when appropriate
        
        The graph is compiled with memory checkpointing for state persistence.
        """
        tool_node = ToolNode(tools=self.available_tools)
        graph_builder = StateGraph(State)
        graph_builder.add_node("call_agent", self.call_agent)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_edge(START, "call_agent")
        graph_builder.add_conditional_edges("call_agent", tools_condition, {"tools": "tools", END: END})
        graph_builder.add_edge("tools", "call_agent")
        self.graph = graph_builder.compile(checkpointer=self.memory)

    def print_message(self, msg: BaseMessage, show_thought: bool):
        """Print a formatted message to the console.
        
        This method formats and displays different types of messages (user, AI, tool)
        with appropriate headers and optional thought process display.
        
        Args:
            msg: The message to display.
            show_thought: Whether to display AI thought process and tool calls.
        """
        if isinstance(msg, HumanMessage):
            print("##" * 10 + " User Message " + "##" * 10)
            print(msg.content + "\n")
        elif isinstance(msg, AIMessage):
            if not msg.tool_calls:
                thought, content = self.extract_ai_thought(msg.content)
                if show_thought and thought:
                    print("##" * 10 + " Thinking... " + "##" * 10)
                    print(thought + "\n")
                print("##" * 10 + " AI Message " + "##" * 10)
                print(content + "\n")
        elif isinstance(msg, ToolMessage):
            if show_thought:
                print("##" * 10 + " Tool Call " + "##" * 10)
                print(msg.content + "\n")
            else:
                pass
        else:
            print("##" * 10 + " Unknown " + "##" * 10)
            print(getattr(msg, 'content', '') + "\n")

    async def stream_graph_updates(self, user_input: str, show_thought: bool = False):
        """Stream and display agent graph updates for user input.
        
        This method processes user input through the agent graph and streams
        the results in real-time, displaying each message as it's generated.
        
        Args:
            user_input: The user's input message to process.
            show_thought: Whether to display AI thought process and tool calls.
            
        Raises:
            RuntimeError: If the agent graph is not built.
        """
        if not self.graph:
            raise RuntimeError("Agent graph is not built. (Most likely a bug.)")
        input_payload = {"messages": [HumanMessage(content=user_input)]}
        async for event in self.graph.astream(input=input_payload, config={"configurable": {"thread_id": "1"}}):
            for value in event.values():
                msg = value["messages"][-1]
                self.print_message(msg, show_thought=show_thought)

    async def cleanup(self):
        """Clean up resources and close connections.
        
        This method properly closes all async context managers and cleans up
        any resources used by the client.
        """
        await self.exit_stack.aclose()


async def start_client(model_provider, model_name, url, server_path, show_thought):
    """Start the MCP client and run the interactive conversation loop.
    
    This function initializes an MCP client, connects to the specified server,
    and runs an interactive loop where users can input messages and receive
    AI responses with optional tool usage.
    
    Args:
        model_provider: The provider for the language model.
        model_name: The specific model name to use.
        url: Optional URL for HTTP-based MCP server connection.
        server_path: Optional path to MCP server script for stdio connection.
        show_thought: Whether to display AI thought process and tool calls.
    """
    client = MCPClient(model_provider=model_provider, model_name=model_name)
    try:
        await client.connect(url=url, server_script_path=server_path)
        while True:
            user_input = input("input: ").strip()
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            await client.stream_graph_updates(user_input, show_thought=show_thought)
    finally:
        await client.cleanup()


def main():
    """Main entry point for the MCP client application.
    
    This function parses command-line arguments, displays configuration information,
    and starts the MCP client with the specified parameters.
    
    Command-line arguments:
        --model-name: The language model name to use (default: qwen3:4b).
        --model-provider: The model provider (default: ollama).
        --url: URL of the MCP HTTP server (mutually exclusive with --server-path).
        --server-path: Path to MCP stdio server script (mutually exclusive with --url).
        --show-thought: Flag to display AI thought process and tool calls.
    """
    parser = argparse.ArgumentParser(description="MCP Client")
    parser.add_argument("--model-name", default="qwen3:4b", help="Model name (default: qwen3:4b)")
    parser.add_argument("--model-provider", default="ollama", help="Provider for the model (default: ollama)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="URL of the MCP HTTP server")
    group.add_argument("--server-path", help="Path to MCP stdio server script (.py or .js)")
    parser.add_argument("--show-thought", action="store_true", help="Show AI's thought process")
    args = parser.parse_args()

    print(f"Model Name: {args.model_name}")
    print(f"Model Provider: {args.model_provider}")
    print(f"Connecting to: {args.url or args.server_path}")
    print(f"Show Thought: {args.show_thought}")

    asyncio.run(start_client(args.model_provider, args.model_name, args.url, args.server_path, args.show_thought))


if __name__ == "__main__":
    main()
