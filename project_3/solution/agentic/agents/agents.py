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
from langgraph.prebuilt import create_react_agent
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

udahub_tools = await mcp_client.get_tools(server_name="udahub")
cultpass_tools = await mcp_client.get_tools(server_name="cultpass")
internal_tools = await mcp_client.get_tools(server_name="internal")
all_tools = await mcp_client.get_tools() + [TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)]

account_access_team_tools_name = ['unblock_user', 'article_rag_search']
customer_inquiry_team_tools_name = ['get_user_subscription_details', 'get_user_reservation_history', 'get_experience_availability', 'article_rag_search']
general_inquiry_team_tools_name = ['article_rag_search', 'web_search']     # Alternative: ['article_rag_search', 'tavily_search_results_json']

account_access_team_tools = [tool for tool in cultpass_tools if tool.name in all_tools ]                            
customer_inquiry_team_tools = [tool for tool in cultpass_tools if tool.name in all_tools ]
general_inquiry_team_tools = [tool for tool in cultpass_tools if tool.name in all_tools ]



# === State Schemas ===
# class TicketState(MessagesState):
#     user_id: str
#     user_status: Literal["active", "cancelled"]
#     user_tier: Literal["basic", "premium"]
#     user_is_blocked: bool = False
#     user_name: str
#     user_email: str  # Note: EmailStr is a Pydantic type; standard TypedDict uses str
#     ticket_content: str
#     ticket_tag: str
#     ticket_urgency: float


# --- Pydantic version ---
class TicketRetrievalState(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    user_id: Annotated[str, Field(description="Unique identifier for the user")]
    account_id: Annotated[str, Field(description="Unique identifier for the account")]     # One company account can have multiple users
    ticket_content: Annotated[str, Field(description="The body of the support request")]
    ticket_tag: Annotated[str, Field(examples=["billing", "technical"])]
    ticket_channel: Annotated[str, Field(description="Channel through which the ticket was submitted, e.g., email, chat, phone")]
    ticket_urgency: Annotated[float, Field(ge=0, le=1, description="Urgency score between 0 and 1")]

class TicketProcessingState(TicketRetrievalState):
    user_status: Annotated[Literal["active", "cancelled"], Field(description="Current account status")]
    user_tier: Annotated[Literal["basic", "premium"], Field(description="Subscription level")]
    user_is_blocked: Annotated[bool, Field(default=False)]
    user_name: Annotated[str, Field(min_length=1)]
    user_email: Annotated[EmailStr, Field(description="Validated user email")]
    ticket_processed: Annotated[bool, Field(default=False)]

class OrchestratorSelection(BaseModel):
    team: Annotated[Literal["customer_inquiry_team", "account_access_team", "general_inquiry_team"], Field(description="Team to route the ticket to")]



# === Agent Creation ===

# --- Ticket Retrieval Agent ---
ticket_retrieval_prompt = ('''
    You are a customer service administrator and you have tools to access the database. 

    Your task is to:
        1. Find the ticket with the highest urgency and retain all the details of the ticket.
        2. Report the details of the ticket, return None if nothing found.
    ''')

ticket_retrieval_agent = create_agent(
    name="ticket_retriever",
    system_prompt=ticket_retrieval_prompt,
    model=llm_small,
    tools=udahub_tools,
    response_format=TicketRetrievalState,
)

# --- User Detail Agent ---
user_detail_prompt = """
    You are a customer database expert and you have tools to access the customer database.
    Your task is to retrieve the user details based on the user_id provided by the user.

    Convert the variables according to the following map:
        - full_name -> user_name
        - is_blocked -> user_is_blocked
        - status -> user_status
        - tier -> user_tier
    """

user_detail_agent = create_agent(
    name="user_detail_agent",
    system_prompt=user_detail_prompt,
    model=llm_small,
    tools=cultpass_tools,
    response_format=TicketProcessingState,
    )

# --- Task Orchestrator Agent ---
task_orchestrator_prompt = """
    You are a expert in sorting customer inquiries and forwarding to the coresponding team, please determine the team based on the following criteria:

    (1) Customer Inquiry Team: If the ticket is related to account cancellation, subscription change, or account reactivation, forward the ticket to the Customer Inquiry Team.
    (2) Account Access Team: If the ticket is related to login issues, password reset, or account lockout, billing questions,forward the ticket to the Account Access Team.
    (3) General Inquiry Team: For any other matters that do not fit the above categories, forward the ticket to the General Inquiry Team.
    (4) If the ticket is 

    Output the team name only without any explanation. The team name should be one of the following: "customer_inquiry_team", "account_access_team", "general_inquiry_team".
    """

task_orchestrator_agent = create_agent(
    name="task_orchestrator_agent",
    system_prompt=task_orchestrator_prompt,
    model=llm_small,
    response_format=OrchestratorSelection,
    )