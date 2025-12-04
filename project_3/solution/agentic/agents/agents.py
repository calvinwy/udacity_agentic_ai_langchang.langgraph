import os
from typing import Dict, Any, List, Literal
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

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm_base_url = "https://openai.vocareum.com/v1"

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    base_url="https://openai.vocareum.com/v1",
    api_key=OPENAI_API_KEY,
)

class TicketState(BaseModel):
    platform: str
    urgency: 
    history: 


def supervisor():



def classifier():



def resolver():


def escalation():
    