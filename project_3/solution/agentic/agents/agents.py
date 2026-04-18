import os
import json
from dotenv import load_dotenv

# LangChain & MCP Imports
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph.message import add_messages

from agentic.state_model import *

load_dotenv()

# ==========================
# === Team Tools Mapping ===
# ==========================

account_access_team_tools_name = ['unblock_user', 'article_rag_search']
customer_inquiry_team_tools_name = ['get_user_subscription_details', 'get_user_reservation_history', 'get_experience_availability', 'article_rag_search']
general_inquiry_team_tools_name = ['article_rag_search', 'web_search']     # Alternative: ['article_rag_search', 'web_search']

# ===========================
# === Model Configuration ===
# ===========================

llm_base_url = "https://api.openai.com/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
UDAHUB_DB_PATH = os.getenv("UDAHUB_DB_PATH")
CULTPASS_DB_PATH = os.getenv("CULTPASS_DB_PATH")
KNOWLEDGE_DB_PATH = os.getenv("KNOWLEDGE_DB_PATH")

llm_small = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, base_url=llm_base_url, api_key=OPENAI_API_KEY)
llm_medium = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, base_url=llm_base_url, api_key=OPENAI_API_KEY)
llm_large = ChatOpenAI(model="gpt-4o", temperature=0.0, base_url=llm_base_url, api_key=OPENAI_API_KEY)

# ============================
# === MCP Helper Functions ===
# ============================
server_configs = {
    "udahub": {
        "transport": "stdio",
        "command": "python",
        "args": ["agentic/tools/udahub_mcp_server.py"],
        "env": {**os.environ, "DATABASE_PATH": os.getenv("UDAHUB_DB_PATH")}
    },
    "cultpass": {
        "transport": "stdio",
        "command": "python",
        "args": ["agentic/tools/cultpass_mcp_server.py"],
        "env": {**os.environ}
    },
    "internal": {
        "transport": "stdio",
        "command": "python",
        "args": ["agentic/tools/internal_mcp_server.py"],
        "env": {**os.environ}
    }
}
mcp_client = MultiServerMCPClient(server_configs)

async def call_tool(tool_list: list, tool_name: str, query: dict = {}):
    my_tool = next((tool for tool in tool_list if tool.name == tool_name), None)
    if not my_tool: return {"error": "tool not found"}
    result = await my_tool.ainvoke(query)
    raw_text = result[0].get("text", "") if isinstance(result, list) else str(result)
    try:
        return json.loads(raw_text) if raw_text.strip() else {}
    except:
        return {"status": "ok", "response": raw_text}

async def initialize_mcp_client():
    '''
    Initialize MCP Connection
    '''
    # Initialize tools
    # cultpass_tools = await mcp_client.get_tools(server_name="cultpass")
    # internal_tools = await mcp_client.get_tools(server_name="internal")
    # tavily_tool = TavilySearch(max_results=3, api_key=os.getenv("TAVILY_API_KEY"))

    # All Tools
    all_tools = await mcp_client.get_tools()
    all_tools = all_tools + [TavilySearch(max_results=3, api_key=TAVILY_API_KEY)]

    return {
        "all_tools": all_tools,
        "mcp_client": mcp_client,
    }

# ============================
# === Agent Initialization ===
# ============================

# --- Agent 1: Task Orchestrator ---
task_orchestrator_prompt = """
You are an expert in sorting customer inquiries and forwarding to the coresponding team, please determine the team based on the following criteria:
(1) Customer Inquiry Team: If the ticket is related to account cancellation, subscription change, or account reactivation, forward the ticket to the Customer Inquiry Team.
(2) Account Access Team: If the ticket is related to login issues, password reset, or account lockout, billing questions, forward the ticket to the Account Access Team.
(3) General Inquiry Team: For any other matters that do not fit the above categories, forward the ticket to the General Inquiry Team.
Output the team name only without any explanation.
"""
task_orchestrator_agent = create_agent(
    name="task_orchestrator_agent",
    system_prompt=task_orchestrator_prompt,
    model=llm_small,
    response_format=OrchestratorSelection,
)

async def initialize_processing_agents(num_customer_inquiry_agents=1, num_account_access_agents=1, num_general_inquiry_agents=1):
    '''
    Wrapping Function Required due to Async of MCP Tools
    '''
    tools_and_client = await initialize_mcp_client()
    all_tools = tools_and_client['all_tools']

    # Tool Mapping
    account_access_team_tools = [tool for tool in all_tools if tool.name in account_access_team_tools_name ]
    customer_inquiry_team_tools = [tool for tool in all_tools if tool.name in customer_inquiry_team_tools_name ]
    general_inquiry_team_tools = [tool for tool in all_tools if tool.name in general_inquiry_team_tools_name ]

    # --- Agent 2: Customer Inquiry Pool ---
    customer_inquiry_team_prompt = """
    You are a customer service representative. Handle issues related to account cancellation, subscription change, or account reactivation.
    Use a considerate and empathetic tone.
    Tools: get_user_subscription_details, get_user_reservation_history, get_experience_availability, article_rag_search.
    """
    customer_inquiry_agents_pool = [
        create_agent(
            name=f"customer_inquiry_agent_{member}",
            system_prompt=customer_inquiry_team_prompt,
            model=llm_medium,
            tools=customer_inquiry_team_tools,
        ) for member in range(1, num_customer_inquiry_agents+1)
    ]

    # --- Agent 3: Account Access Pool ---
    account_access_team_prompt = """
    You are a customer service representative. Handle issues related to login, password reset, account lockout, and billing.
    Use a considerate and empathetic tone.
    Tools: unblock_user, article_rag_search.
    """
    account_access_agent_pool = [
        create_agent(
            name=f"account_access_agent_{member}",
            system_prompt=account_access_team_prompt,
            model=llm_medium,
            tools=account_access_team_tools,
        ) for member in range(1, num_account_access_agents+1)
    ]

    # --- Agent 4: General Inquiry Pool ---
    general_inquiry_team_prompt = """
    You are a customer service representative. Handle general issues outside of account access and customer booking.
    Tools: article_rag_search, web_search.
    """
    general_inquiry_agent_pool = [
        create_agent(
            name=f"general_inquiry_agent_{member}",
            system_prompt=general_inquiry_team_prompt,
            model=llm_medium,
            tools=general_inquiry_team_tools,
        ) for member in range(1, num_general_inquiry_agents+1)
    ]

    # --- Agent 5: Confidence Evaluator ---
    confidence_eval_prompt = """
    You are a customer service quality evaluation consultant. Evaluate the quality of the answer with a score from 0 to 1.
    """
    confidence_eval_agent = create_agent(
        name="confidence_evaluator",
        system_prompt=confidence_eval_prompt,
        model=llm_large,
        response_format=ConfidentScore,
    )

    return {
        "customer_inquiry_agents_pool": customer_inquiry_agents_pool,
        "account_access_agents_pool": account_access_agent_pool,
        "general_inquiry_agent_pool": general_inquiry_agent_pool,
        "evaluator": confidence_eval_agent,
    }
