import os
import asyncio
import sqlite3

from agentic.state_model import *
from agentic.agents.agents import *
from agentic.workflow import *
from agentic.tools.tools import *

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, filter_messages

# =========================
# === System Parameters ===
# =========================
# System Parameters
recursion_limit = 100   # Currently not used, for previous debugging
# Architecture Parameters
num_customer_inquiry_agents = 3
num_account_access_agents = 2
num_general_inquiry_agents = 3
escalation_threshold = 0.7

# ==================
# === MCP Config ===
# ==================
# server_configs = {
#     "udahub": {
#         "transport": "stdio",
#         "command": "python",
#         "args": ["agentic/tools/udahub_mcp_server.py"],
#         "env": {**os.environ, "DATABASE_PATH": os.getenv("UDAHUB_DB_PATH")}
#     },
#     "cultpass": {
#         "transport": "stdio",
#         "command": "python",
#         "args": ["agentic/tools/cultpass_mcp_server.py"],
#         "env": {**os.environ}
#     },
#     "internal": {
#         "transport": "stdio",
#         "command": "python",
#         "args": ["agentic/tools/internal_mcp_server.py"],
#         "env": {**os.environ}
#     }
# }
# mcp_client = MultiServerMCPClient(server_configs)

# =========================
# === Chatbot Interface ===
# =========================

# Deterministic Ticket Extraction Procedure
async def process_ticket(ticket_processing_graph, config):
    print("Processing tickets from Udahub database...")

    # Obtain Highest Priority Ticket
    # --- Using tool.py ---
    response = get_highest_urgency_ticket()
    print(response)

    # --- Using MCP Tool ---
    # tools_and_client = await initialize_mcp_client()
    # all_tools = tools_and_client['all_tools']
    # tool_name = "get_highest_urgency_ticket"
    # response = await call_tool(all_tools, tool_name)
    # print(response)

    # Evaluate Ticket
    if response.status == 'not_found':
        print("<There is no more tickets in the queue to process.>")
        success = False
    else:
        ticket = response.ticket.model_dump()
        ticket_content = ticket['ticket_content']
        print(f"Ticket: {ticket_content}")

        config['thread_id'] = ticket['user_id']

        process_response = await ticket_processing_graph.ainvoke(input={"messages": "Start processing tickets.", **ticket}, config=config)
        response_ai = filter_messages(
            process_response["messages"],
            include_types=[AIMessage],
            )
        response_content = response_ai[-1].content
        print(f"AI: {response_content}")
        success = True
    return {
        'success': success,
        'user_id': ticket['user_id'],
    }

async def run():
    '''
    Main Routine of Chatbot
    '''
    # ---------------------------
    # --- Initialize Subgraph ---
    # ---------------------------
    teams_subgraph = await create_teams_graphs(
        num_customer_inquiry_agents, 
        num_account_access_agents, 
        num_general_inquiry_agents, 
        escalation_threshold
    )
    agent_swarm_map = teams_subgraph["agent_swarm_map"]
    agent_teams = teams_subgraph["agent_teams"]

    # ----------------------
    # --- Graph Creation ---
    # ----------------------
    # Config required for RR Agent Swarm Creation
    config = {
        "configurable": {
            "agent_teams": agent_teams,
            "agent_swarm_map": agent_swarm_map,
            "recursion_limit": recursion_limit,
            "store": store,
        }
    }
    ticket_processing_graph = await create_main_graph(config)

    # ---------------------
    # --- Start Chatbot ---
    # ---------------------

    def menu():
        user_id = input("Please enter your User ID: ").strip().lower()
        if not user_id:
            user_id = "default_user"
        
        print('''
            Welcome to ChatService, you have the following options:
                    (1) Type 'quit' or 'exit' if you want to leave anytime.
                    (2) Press ENTER without any input to process the next ticket in the queue.
                    (3) Type 'all' to process all tickets in the queue.
                    (4) Type your query directly to get response from the chatbot.
            ''')
        print("Please type your question:")
        user_input = input(">> ").strip().lower()
        return {
            'user_id': user_id,
            'user_input': user_input
        }

    # Chatbot Loop
    while True:
        try:
            menu_input = menu()                        
            user_input = menu_input['user_input']

            # --- Different Configuration for Different Mode ---
            config_live_chat = {
                "configurable": {
                    "thread_id": menu_input['user_id'],
                    "agent_teams": agent_teams,
                    "agent_swarm_map": agent_swarm_map,
                    "recursion_limit": recursion_limit,
                    "store": store,
                }
            }
            config_load_ticket = {
                "configurable": {
                    "thread_id": None,
                    "agent_teams": agent_teams,
                    "agent_swarm_map": agent_swarm_map,
                    "recursion_limit": recursion_limit,
                    "store": store,
                }
            }

            # --- Mode Decision ---
            if user_input in ["quit", "exit"]:
                print("Exiting...")
                break
            elif user_input == "all":
                while True:
                    response = await process_ticket(ticket_processing_graph, config_load_ticket)
                    if response['success'] == False:
                        break
            elif len(user_input) == 0:
                while True:
                    response = await process_ticket(ticket_processing_graph, config_load_ticket)
                    config_continue = {
                        "configurable": {
                            "thread_id": response['user_id'],
                            "agent_teams": agent_teams,
                            "agent_swarm_map": agent_swarm_map,
                            "recursion_limit": recursion_limit,
                            "store": store,
                        }
                    }
                    while True:
                        try:
                            user_input = input(">> ").strip().lower()
                            if user_input in ["quit", "exit"]:
                                print("Exiting...")
                                break
                            else:
                                input_state = {
                                    "user_id": response['user_id'],
                                    "ticket_content": user_input,
                                    "ticket_id": "live_session",
                                    "ticket_channel": "live_chatbot",
                                    "ticket_urgency": 1,
                                }
                                result = await ticket_processing_graph.ainvoke(input=input_state, config=config_continue)
                        except KeyboardInterrupt:
                            print("Exiting...")
                            break
            else:
                input_state = {
                    "user_id": menu_input['user_id'],
                    "ticket_content": user_input,
                    "ticket_id": "live_session",
                    "ticket_channel": "live_chatbot",
                    "ticket_urgency": 1,
                }
                result = await ticket_processing_graph.ainvoke(input=input_state, config=config_live_chat)
                print(result["messages"].content)
        except KeyboardInterrupt:
            print("Exiting...")
            break

def main() -> None:
    asyncio.run(run())

if __name__ == "__main__":
    main()