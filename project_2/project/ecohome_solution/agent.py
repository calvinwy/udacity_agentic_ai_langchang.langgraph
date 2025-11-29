import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from langchain_core.messages import SystemMessage
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from tools import TOOL_KIT


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm_base_url = "https://openai.vocareum.com/v1"

class Agent:

    def __init__(self, instructions:str, model:str="gpt-4o-mini"):
    # def __init__(self, instructions:str, model:str="gpt-4o"):

        # Initialize the LLM
        llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            base_url="https://openai.vocareum.com/v1",
            api_key=OPENAI_API_KEY,
        )

        # Create the Energy Advisor agent
        self.graph = create_react_agent(
            name="energy_advisor",
            prompt=SystemMessage(content=instructions),
            model=llm,
            tools=TOOL_KIT,
        )

    def invoke(self, question: str, context:str=None) -> str:
        """
        Ask the Energy Advisor a question about energy optimization.
        
        Args:
            question (str): The user's question about energy optimization
            context (str): Relevant contextual information related to the question
        
        Returns:
            str: The advisor's response with recommendations
        """
        
        messages = []
        if context:
            # Add some context to the question as a system message
            messages.append(
                ("system", context)
            )

        messages.append(
            ("user", question)
        )
        
        # Get response from the agent
        response = self.graph.invoke(
            input= {
                "messages": messages
            }
        )
        
        return response

    def get_agent_tools(self):
        """Get list of available tools for the Energy Advisor"""
        return [t.name for t in TOOL_KIT]