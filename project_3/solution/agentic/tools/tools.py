import os
from typing import Any, Dict, Optional
from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from typing import Annotated, Optional, Literal, List, Dict, TypedDict
from pydantic import BaseModel

# class HighestUrgencyTicket(TypedDict, total=False):     # `total=False`: None of these keys are strictly required
#     user_id: str
#     account_id: str
#     ticket_id: str
#     ticket_content: str
#     ticket_tag: str
#     ticket_channel: str
#     ticket_urgency: float

class TicketInfo(BaseModel):
    user_id: str
    account_id: str
    ticket_id: str
    ticket_content: str | None = None
    ticket_channel: str | None = None
    ticket_tag: str | None = None
    ticket_urgency: float

class HighestUrgencyTicketResult(BaseModel):
    status: Literal["ok", "not_found"]
    ticket: TicketInfo | None = None
    message: str | None = None

# --- Configuration ---
load_dotenv()
UDAHUB_DB_PATH = os.getenv("UDAHUB_DB_PATH", "data/core/udahub.db")
DB_URL = f"sqlite:///{UDAHUB_DB_PATH}"
engine = create_engine(DB_URL)

@tool()
def get_highest_urgency_ticket() -> HighestUrgencyTicket:
    """
    Fetches the ticket with the highest urgency score using raw SQL.
    """
    query = text("""
        SELECT 
            t.ticket_id, 
            tm.content, 
            u.external_user_id AS owner_id, 
            t.channel, 
            tmd.tags, 
            t.account_id, 
            tmd.urgency_score
        FROM tickets t
        JOIN ticket_metadata tmd ON t.ticket_id = tmd.ticket_id
        JOIN users u ON t.user_id = u.user_id
        JOIN ticket_messages tm ON t.ticket_id = tm.ticket_id
        ORDER BY tmd.urgency_score DESC
        LIMIT 1
    """)

    try:
        with engine.connect() as conn:
            row = conn.execute(query).mappings().first()
        if row is None:
            return HighestUrgencyTicketResult(
                status="not_found",
                message="No tickets found",
            )
        return HighestUrgencyTicketResult(
            status="ok",
            ticket=TicketInfo(
                ticket_id=row["ticket_id"],
                ticket_content=row["content"],
                user_id=row["owner_id"],
                ticket_channel=row["channel"],
                ticket_tag=row["tags"],
                account_id=row["account_id"],
                ticket_urgency=row["urgency_score"],
            ),
        )
    except Exception as e:
        raise SystemError(f"Failed to fetch highest urgency ticket: {e}")

@tool()
def delete_ticket(ticket_id: str) -> str:
    """
    Removes a ticket and its associated child records using raw SQL.
    """
    # Since SQLite might not have PRAGMA foreign_keys = ON by default,
    # we manually delete from child tables to be safe.
    queries = [
        text("DELETE FROM ticket_messages WHERE ticket_id = :tid"),
        text("DELETE FROM ticket_metadata WHERE ticket_id = :tid"),
        text("DELETE FROM tickets WHERE ticket_id = :tid")
    ]

    try:
        with engine.begin() as conn:
            for q in queries:
                conn.execute(q, {"tid": ticket_id})
        return f"Success: Ticket {ticket_id} and associated data deleted."
    except Exception as e:
        raise SystemError(f"Failed to fetch highest urgency ticket: {e}")