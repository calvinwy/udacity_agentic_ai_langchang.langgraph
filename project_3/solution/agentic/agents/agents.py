import os
from typing import Annotated, Literal, TypedDict
from IPython.display import Image, display
from pydantic import BaseModel, Field
import nest_asyncio
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
UDAHUB_DB_PATH = os.getenv("UDAHUB_DB_PATH")
CULTPASS_DB_PATH = os.getenv("CULTPASS_DB_PATH")
KNOWLEDGE_DB_PATH = os.getenv("KNOWLEDGE_DB_PATH")
llm_base_url = "https://api.openai.com/v1"


# === LLM and Embeddings Setup ===
llm_small = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.0,
    base_url=llm_base_url,
    api_key=OPENAI_API_KEY,
)

llm_medium = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    base_url=llm_base_url,
    api_key=OPENAI_API_KEY,
)

llm_large = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    base_url=llm_base_url,
    api_key=OPENAI_API_KEY,
)

embeddings_fn = OpenAIEmbeddings(
    model="text-embedding-3-large",
    base_url=llm_base_url,
    api_key=OPENAI_API_KEY,
)



# === MCP Servers and Tools ===
# Define the configurations for all servers
server_configs = {
    "udahub": {
        "transport": "stdio",
        "command": "python",
        "args": ["agentic/tools/udahub_mcp_server.py"],
        "env": {
            **os.environ,          # Inherit existing environment variables
            "DATABASE_PATH": UDAHUB_DB_PATH
        }
    },
    "cultpass": {
        "transport": "stdio",
        "command": "python",
        "args": ["agentic/tools/cultpass_mcp_server.py"],
        "env": {
            **os.environ,          # Inherit existing environment variables
        }
    },
    "internal": {
        "transport": "stdio",
        "command": "python",
        "args": ["agentic/tools/internal_mcp_server.py"],
        "env": {
            **os.environ,          # Inherit existing environment variables
        }
    }
}

mcp_client = MultiServerMCPClient(server_configs)

