import os
import json
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
UDAHUB_DB_PATH = os.getenv("UDAHUB_DB_PATH")
CULTPASS_DB_PATH = os.getenv("CULTPASS_DB_PATH")
KNOWLEDGE_DB_PATH = os.getenv("KNOWLEDGE_DB_PATH")
llm_base_url = "https://api.openai.com/v1"


# ===========================
# === Model Configuration ===
# ===========================

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

# ========================
# === MCP Client Setup ===
# ========================

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

udahub_tools = mcp_client.get_tools(server_name="udahub")
cultpass_tools = mcp_client.get_tools(server_name="cultpass")
internal_tools = mcp_client.get_tools(server_name="internal")
all_tools = mcp_client.get_tools() + [TavilySearch(max_results=3, api_key=TAVILY_API_KEY)]

account_access_team_tools_name = ['unblock_user', 'article_rag_search']
customer_inquiry_team_tools_name = ['get_user_subscription_details', 'get_user_reservation_history', 'get_experience_availability', 'article_rag_search']
general_inquiry_team_tools_name = ['article_rag_search', 'web_search']     # Alternative: ['article_rag_search', 'web_search']

account_access_team_tools = [tool for tool in cultpass_tools if tool.name in all_tools ]                            
customer_inquiry_team_tools = [tool for tool in cultpass_tools if tool.name in all_tools ]
general_inquiry_team_tools = [tool for tool in cultpass_tools if tool.name in all_tools ]

# Direct MCP tool calling function
async def call_tool(tool_list: list, tool_name: str, query: dict = {}):
    my_tool = next(tool for tool in tool_list if tool.name == tool_name)
    result = await my_tool.ainvoke(query)
    raw_text = result[0].get("text", "")
    if not raw_text.strip():
        return {} # Return empty dict if there's no content
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # If it's not JSON, return the raw text or a status dict
        return {"status": "ok", "response": raw_text}
    
all_tools_name = [ tool.name for tool in all_tools ]

# ==============
# === Agents ===
# ==============

task_orchestrator_prompt = """
    You are an expert in sorting customer inquiries and forwarding to the coresponding team, please determine the team based on the following criteria:

    (1) Customer Inquiry Team: If the ticket is related to account cancellation, subscription change, or account reactivation, forward the ticket to the Customer Inquiry Team.
    (2) Account Access Team: If the ticket is related to login issues, password reset, or account lockout, billing questions, forward the ticket to the Account Access Team.
    (3) General Inquiry Team: For any other matters that do not fit the above categories, forward the ticket to the General Inquiry Team.

    Output the team name only without any explanation. The team name should be one of the following: "customer_inquiry_team", "account_access_team", "general_inquiry_team".
    """

task_orchestrator_agent = create_agent(
    name="task_orchestrator_agent",
    system_prompt=task_orchestrator_prompt,
    model=llm_small,
    response_format=OrchestratorSelection,
    )

customer_inquiry_team_prompt = """
    You are a customer service representative.
    You are given the customer's inquiry ticket.  Your task is to address issues related to account cancellation, subscription change, or account reactivation.
    Please use a considerate and empathetic tone when communicating with customers, acknowledging their concerns and providing clear, helpful information.

    You are given a set of tools to address these issues:
        get_user_subscription_details: Retrieve the user's current subscription details, including subscription tier and renewal date.
        get_user_reservation_history: Fetch the user's past reservation history, including dates, experiences booked, and any cancellations.
        get_experience_availability: Check the availability of specific experiences that the user is interested in, based on their preferences and past bookings.
        article_rag_search: Search the knowledge base for relevant articles that can assist in resolving the customer's inquiry, providing summaries and links to the articles when applicable.

    If you cannot complete the task with the tools above, please stop and explain the reason.
    If you do not understand the ticket, please ask the user to clarify the inquiry.
    """

customer_inquiry_agents_pool = [
    create_agent(
        name=f"customer_inquiry_agent_{member}",
        system_prompt=customer_inquiry_team_prompt,
        model=llm_medium,
        tools=customer_inquiry_team_tools,
    ) for member in range(1, num_customer_inquiry_agents+1)
]

account_access_team_prompt = """
    You are a customer service representative.
    You are given the customer's inquiry ticket.  Your task is to address issues related to login issues, password reset, or account lockout, billing questions.
    Please use a considerate and empathetic tone when communicating with customers, acknowledging their concerns and providing clear, helpful information.

    You are given a set of tools to address these issues:
        unblock_user: Unblock a user's account if they are locked out due to multiple failed login attempts or other security reasons.
        article_rag_search: Search the knowledge base for relevant articles that can assist in resolving the customer's inquiry, providing summaries and links to the articles when applicable.

    If you cannot complete the task with the tools above, please stop and explain the reason.
    If you do not understand the ticket, please ask the user to clarify the inquiry.
    """

account_access_agent_pool = [
    create_agent(
        name=f"account_access_agent_{member}",
        system_prompt=customer_inquiry_team_prompt,
        model=llm_medium,
        tools=account_access_team_tools,
        ) for member in range(1, num_account_access_agents+1)
]

general_inquiry_team_prompt = """
    You are a customer service representative.
    You are given the customer's inquiry ticket.  Your task is to address general issues outisde of account access, customer booking, and billing questions.
    Please use a considerate and empathetic tone when communicating with customers, acknowledging their concerns and providing clear, helpful information.

    You are given a set of tools to address these issues:
        article_rag_search: Search the knowledge base for relevant articles that can assist in resolving the customer's inquiry, providing summaries and links to the articles when applicable.
        web_search: Perform a web search to find up-to-date information that can help address the customer's inquiry, summarizing key findings and providing relevant links.

    If you cannot complete the task with the tools above, please stop and explain the reason.
    If you do not understand the ticket, please ask the user to clarify the inquiry.
    """

general_inquiry_agent_pool = [
    create_agent(
        name=f"general_inquiry_agent_{member}",
        system_prompt=general_inquiry_team_prompt,
        model=llm_medium,
        tools=general_inquiry_team_tools,
        ) for member in range(1, num_general_inquiry_agents+1)
]

class ConfidentScore(BaseModel):
    confident_score: Annotated[Optional[float], Field(ge=0, le=1, default=0, description="Confidence score for RAG and Web search output.")]

confidence_eval_prompt = """
    You are a customer service quality evaluation consultant, you are tasked to evaluate the quality of the
    customer service team's answer with a score from 0 to 1 where 0 is bad and 1 is good.  Only provide the
    score and nothing else.

    Output:
        confident_score: Annotated[Optional[float], Field(ge=0, le=1, default=0, description="Confidence score for RAG and Web search output.")]
    """

confidence_eval_agent = create_agent(
        name=f"confidence_evalator",
        system_prompt=confidence_eval_prompt,
        model=llm_large,
        response_format=ConfidentScore,
)