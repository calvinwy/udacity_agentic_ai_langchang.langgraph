import json
import sqlite3

from typing import Annotated, Optional, Literal, List, Dict, TypedDict

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.message import MessagesState, add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, filter_messages
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from agentic.state_model import *
from agentic.agents.agents import *

# =========================
# === Memory Management ===
# =========================

db_path = "./data/core/runtime_memory.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
checkpointer = SqliteSaver(conn)
store = InMemoryStore()

# ================================
# === Round-Robin Teams Graphs ===
# ================================

async def create_teams_graphs(num_customer_inquiry_agents, num_account_access_agents, num_general_inquiry_agents, escalation_threshold) -> dict:
    '''
    Wrapper Function Required for Asynchronous Processing
    '''

    # ----------------------------
    # --- Initialize MCP Tools ---
    # ----------------------------
    agents_and_tools = await initialize_processing_agents(num_customer_inquiry_agents, num_account_access_agents, num_general_inquiry_agents)

    customer_inquiry_agents_pool = agents_and_tools['customer_inquiry_agents_pool']
    account_access_agents_pool = agents_and_tools['account_access_agents_pool']
    general_inquiry_agent_pool = agents_and_tools['general_inquiry_agent_pool']
    confidence_eval_agent = agents_and_tools['evaluator']

    agent_swarm_map = {
        "customer_inquiry_team": customer_inquiry_agents_pool,
        "account_access_team": account_access_agents_pool,
        "general_inquiry_team": general_inquiry_agent_pool,
    }

    # ------------------------
    # --- Node Definitions ---
    # ------------------------

    confidence_eval_template = PromptTemplate.from_template(
        """
        Customer's question: 
        {question}

        Customer service team's answer: 
        {answer}
        """
    )

    async def confident_evaluator_node(state: RoundRobinState) -> Command[Literal["human_escalation", END]]:
        '''
        Evaluate the confidence level of the output
        '''
        print("Evaluating the response quality...")

        formatted_input = confidence_eval_template.format(question=state.ticket_content, answer=state.messages[-1].content)
        result = await confidence_eval_agent.ainvoke(input = {"messages":[("user", formatted_input)]})
        
        # Convert the confidence message into SystemMessage
        # (Otherwise, system would misunderstood it as message response.)
        last_message = result['messages'][-1]
        if isinstance(last_message, AIMessage):
            converted_message = SystemMessage(content=last_message.content)
        else:
            converted_message = []

        confident_score = result['structured_response'].confident_score
        next_destination = END if confident_score > escalation_threshold else "human_escalation"

        return Command(
            update={
                "messages": converted_message,
                "confident_score": confident_score,
                },
                goto=next_destination
        )

    def human_escalation_node(state: RoundRobinState) -> dict:  
        '''
        Escalate to Human
            Description:
                Mark the ticket for human review in the database
        '''
        print('Updating ticket status for human agent to process later...')
        return {'ticket_status': 'Human'}
    
    def update_index_node(state: RoundRobinState):
        '''
        Index Update Node at the Graph Entrance
        '''
        state = state.model_dump()
        agent_names = state["agent_names"]
        current_agent_index = state.get("current_agent_index", 0)
        current_agent_index = 0 if current_agent_index is None else current_agent_index
        new_agent_index = current_agent_index + 1  
        new_agent_index_in_range = new_agent_index % len(agent_names)
        print(f"new_agent_index_in_range: {new_agent_index_in_range}")
        return {
            "current_agent_index": new_agent_index_in_range
        }

    def route_round_robin(state: RoundRobinState):
        '''
        Routing Function for Round-Robin Agents
        '''
        state = state.model_dump()
        agent_names = state["agent_names"]
        current_agent_index = state.get("current_agent_index", 0)
        current_agent_index = 0 if current_agent_index is None else current_agent_index
        current_agent_index_in_range = current_agent_index % len(agent_names)
        active_agent = agent_names[current_agent_index_in_range]
        print("Round-robin Active Agent:", active_agent)
        return active_agent

    # ---------------------------
    # --- Workflow Definition ---
    # ---------------------------

    def create_rr_team_workflow(name:str, agent_pool: List[CompiledStateGraph]):
        '''
        Building Graph for a Single Team of Multiple RR Agents

            Args:
                name: The name given to the created team graph
                agent_pool: The dictionary wit the member agents of this team (key: agent name, value: compiled graph agent)
            
            Returns:
                workflow: StateGraph type for the workflow to be modified with post-processing step
        '''
        workflow = StateGraph(RoundRobinState)

        workflow.add_node("update_index", update_index_node)
        for agent in agent_pool:
            workflow.add_node(agent.name, agent)

        workflow.add_edge(START, "update_index")
        workflow.add_conditional_edges(
            source="update_index",
            path=route_round_robin,
            path_map=[agent.name for agent in agent_pool]
        )
        return workflow

    # ----------------------
    # --- Compile Graphs ---
    # ----------------------

    # --- List of Graph for Each Team in RR Subgraph ---
    # (Global Variables use by Agents)
    agent_teams_workflow = { team_name: create_rr_team_workflow(team_name, agent_pool) for team_name, agent_pool in agent_swarm_map.items() }

    team = 'general_inquiry_team'
    modify_workflow = agent_teams_workflow[team]
    modify_workflow.add_node("confident_evaluator", confident_evaluator_node)
    modify_workflow.add_node("human_escalation", human_escalation_node)
    for agent in agent_swarm_map[team]:
        modify_workflow.add_edge(agent.name, "confident_evaluator")

    agent_teams_workflow[team] = modify_workflow
    agent_teams = { key: value.compile(name=key, checkpointer=checkpointer, store=store) for key, value in agent_teams_workflow.items() }

    return {
        'agent_swarm_map': agent_swarm_map,
        'agent_teams': agent_teams,
    }

# ======================================
# === Ticket Processing (Main Graph) ===
# ======================================

async def create_main_graph(config: dict):

    tools_and_client = await initialize_mcp_client()
    all_tools = tools_and_client['all_tools']

    # ------------------------
    # --- Node Definitions ---
    # ------------------------

    async def user_detail_node(ticket: TicketRetrievalState) -> dict:
        """
        Extract User Information based on the user_id of the current ticket

            Args:
                ticket: The current ticket to process.
            
            Returns:
                dict: Conforming to the `TicketProcessingState` Pydantic schema, returning the information
                    combining the ticket and user details.
        """
        
        print("Retrieving user information...")

        # Retrieving high priority ticket
        tool_name = "get_user_subscription_details"
        user_detail = await call_tool(all_tools, tool_name, {"user_id": ticket.user_id})

        # Erasing the retrieved ticket
        if len(user_detail) == 0:     # Empty dictionary == User not found
            out_message = "User not found."
        else:
            out_message = f"User details retrieved successfully: \n{json.dumps(user_detail)}"

        # Retrieving chat history
        history = await call_tool(all_tools, "get_user_ticket_history", {"external_user_id": ticket.user_id})
        if history:
            history_str = "\n".join([f"- [{h['status']}] {h['message'][:100]}" for h in history])
            out_message += f"\n\nPrevious interaction history:\n{history_str}"

        print(out_message)
        output = {
                "messages": [SystemMessage(content=out_message)],
                **user_detail,    
            }
        return output


    async def task_orchestrator_node(ticket: TicketProcessingState) -> Command[Literal["trigger_agent_team", END]]:
        '''
        Evaluate the ticket content to route the ticket to the corresponding team.
        '''
        # Being Process Ticket if Status is `Open`
        if ticket.ticket_status.lower() == 'open':
            output = await task_orchestrator_agent.ainvoke({"messages": [{"role": "system", "content": f"ticket_content: {ticket.ticket_content}"}]})
            messages = output["messages"]
            next_team = output['structured_response'].team
            next_destination = "trigger_agent_team"
        # Finish Process Ticket if Statu is `Closed` or `Human`
        else:
            next_team = None
            next_destination = END
            messages = []

            # Update ticket status in the Udahub database after processing the ticket
            tool_name = "update_ticket_status"
            _ = await call_tool(all_tools, tool_name, {"ticket_id": ticket.ticket_id, "new_status": ticket.ticket_status.lower()})

        return Command(
            update={
                "messages": messages,
                "ticket_status": ticket.ticket_status,
                "team_to_call": next_team,
                },
                goto=next_destination
        )

    async def trigger_agent_team_node(state: TicketProcessingState, config: RunnableConfig):
        '''
        Node Contain Subgraph for Multiple Ticket Processing Agent Teams
        '''
        team_to_call = state.team_to_call

        agent_teams: dict[str,CompiledStateGraph] = config["configurable"]["agent_teams"]
        agent_swarm_map = config["configurable"]["agent_swarm_map"]
        agent_pool_to_call: List[CompiledStateGraph] = agent_swarm_map[team_to_call]

        if team_to_call in agent_teams.keys():
            team = agent_teams[team_to_call]
            agent_names = [agent.name for agent in agent_pool_to_call]
            # print(f"Agent Names: {agent_names}")
            agent_input = state.model_dump()
            agent_input['agent_names'] = agent_names
            # print(f"Agent Input: {agent_input}")
            agent_input = RoundRobinState(**agent_input)
            result = await team.ainvoke(
                input=agent_input,
                config={
                    "configurable": {
                        "thread_id": "round_robin"
                    }
                }
            )
            result['ticket_status'] = 'closed' if result['ticket_status'] == 'open' else result['ticket_status']
            return result
        else:
            raise ValueError("team_to_call is not inside agent_teams")  

    # --- Main Graph ---
    workflow = StateGraph(TicketProcessingState)

    workflow.add_node("user_detail", user_detail_node)
    workflow.add_node("task_orchestrator", task_orchestrator_node)
    workflow.add_node("trigger_agent_team", trigger_agent_team_node)

    workflow.add_edge(START, "user_detail")
    workflow.add_edge("user_detail", "task_orchestrator")
    workflow.add_edge("trigger_agent_team", "task_orchestrator")

    ticket_processing_graph = workflow.compile(checkpointer=checkpointer, store=store)

    # ------------------------------
    # --- Return Compiled Graphs ---
    # ------------------------------
    return ticket_processing_graph


