import os
from typing import Any, Dict, Optional
from mcp.server.fastmcp import FastMCP
from sqlalchemy import create_engine, text

from typing import TypedDict
from mcp.types import CallToolResult

from typing import Annotated, Optional, Literal, List, Dict, TypedDict
from pydantic import BaseModel
from fastmcp.exceptions import ToolError

# class HighestUrgencyTicket(TypedDict, total=False):     # `total=False`: None of these keys are strictly required
#     ticket_id: str
#     user_id: str
#     account_id: str
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
    ticket_status: str | None = None

class HighestUrgencyTicketResult(BaseModel):
    status: Literal["ok", "not_found"]
    ticket: TicketInfo | None = None
    message: str | None = None

# --- Configuration ---
# UDAHUB_DB_PATH = os.path.join(os.getcwd(), os.getenv("UDAHUB_DB_PATH", "data/core/udahub.db"))
UDAHUB_DB_PATH = os.getenv("UDAHUB_DB_PATH", "data/core/udahub.db")
DB_URL = f"sqlite:///{UDAHUB_DB_PATH}"
engine = create_engine(DB_URL)

# Initialize MCP Server
mcp = FastMCP("Udahub Ticket Manager")

@mcp.tool()
def get_highest_urgency_ticket() -> HighestUrgencyTicketResult:
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
            tmd.urgency_score,
            tmd.status
        FROM tickets t
        JOIN ticket_metadata tmd ON t.ticket_id = tmd.ticket_id
        JOIN users u ON t.user_id = u.user_id
        JOIN ticket_messages tm ON t.ticket_id = tm.ticket_id
        WHERE tmd.status = 'open'
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
                ticket_status=row["status"],
            ),
        )
    except Exception as e:
        raise SystemError(f"Failed to fetch highest urgency ticket: {e}")


@mcp.tool()
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
        raise ToolError(f"Error: {str(e)}")

@mcp.tool()
def update_ticket_status(ticket_id: str, new_status: str) -> str:
    """
    Updates the status of a ticket using raw SQL.
    
    Args:
        ticket_id: The ID of the ticket to update.
        new_status: The new status string.
    """
    # SQL query to update the metadata table
    # We use :variable syntax to prevent SQL injection
    sql_query = text("""
        UPDATE ticket_metadata 
        SET status = :status, 
            updated_at = CURRENT_TIMESTAMP
        WHERE ticket_id = :tid
    """)

    try:
        with engine.begin() as conn:
            result = conn.execute(sql_query, {"status": new_status, "tid": ticket_id})
            
            if result.rowcount == 0:
                return f"No ticket found with ID: {ticket_id}"
            
            return f"Successfully updated ticket {ticket_id} to status: {new_status}"
            
    except Exception as e:
        return f"Database error: {str(e)}"

if __name__ == "__main__":
    mcp.run()