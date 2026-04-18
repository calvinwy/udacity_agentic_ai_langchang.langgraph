from typing import Annotated, Optional, List, Literal
from pydantic import BaseModel, Field, EmailStr
from langgraph.graph.message import add_messages

class TicketRetrievalState(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    user_id: Annotated[Optional[str], Field(description="Unique identifier for the user", default=None)]
    account_id: Annotated[Optional[str], Field(description="Unique identifier for the account", default=None)]
    ticket_id: Annotated[Optional[str], Field(description="Unique identifier for the ticket", default=None)]
    ticket_content: Annotated[Optional[str], Field(description="The body of the support request", default=None)]
    ticket_tag: Annotated[Optional[str], Field(examples=["billing", "technical"], default=None)]
    ticket_channel: Annotated[Optional[str], Field(description="Channel through which the ticket was submitted", default=None)]
    ticket_urgency: Annotated[Optional[float], Field(ge=0, le=1, description="Urgency score", default=None)]
    ticket_status: Annotated[Optional[str], Field(description="Current status of the ticket", default='open')]

class TicketProcessingState(TicketRetrievalState):
    user_status: Annotated[Optional[Literal["active", "cancelled"]], Field(description="Current account status", default=None)]
    user_tier: Annotated[Optional[Literal["basic", "premium"]], Field(description="Subscription level", default=None)]
    user_is_blocked: Annotated[Optional[bool], Field(default=False)]
    user_name: Annotated[Optional[str], Field(default=None)]
    user_email: Annotated[Optional[EmailStr], Field(description="Validated user email", default=None)]
    team_to_call: Annotated[Optional[str], Field(default=None)]

class RoundRobinState(TicketProcessingState):
    agent_names: List[str] = Field(default_factory=list)
    current_agent_index: Annotated[Optional[int], Field(description="Round-Robin turn index", default=None)]
    confident_score: Annotated[Optional[float], Field(ge=0, le=1, default=0, description="Confidence score for RAG and Web search output.")]

class OrchestratorSelection(BaseModel):
    team: Annotated[Literal["customer_inquiry_team", "account_access_team", "general_inquiry_team"], Field(description="Team to route the ticket to")]

class ConfidentScore(BaseModel):
    confident_score: Annotated[Optional[float], Field(ge=0, le=1, default=0, description="Confidence score for quality evaluation.")]