import os
from typing import Any, Dict, Optional
from mcp.server.fastmcp import FastMCP
from sqlalchemy import create_engine, text

# --- Configuration ---
UDAHUB_DB_PATH = os.getenv("UDAHUB_DB_PATH", "data/core/udahub.db")
DB_URL = f"sqlite:///{UDAHUB_DB_PATH}"
engine = create_engine(DB_URL)

# Initialize MCP Server
mcp = FastMCP("Udahub Ticket Manager")

@mcp.tool()
def get_highest_urgency_ticket() -> Optional[Dict[str, Any]]:
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

    with engine.connect() as conn:
        result = conn.execute(query).mappings().first()
        
        if not result:
            return {}
            # return {"error": "User or subscription not found."}
        else:
            result = dict(result)
            result["user_id"] = result.pop("owner_id")
            result["account_id"] = result.pop("account_id")
            result["ticket_content"] = result.pop("content")
            result["ticket_tag"] = bool(result.pop("tags"))
            result["ticket_channel"] = result.pop("channel")
            result["ticket_urgency"] = result.pop("urgency_score")
            return result

# @mcp.tool()
# def delete_ticket(ticket_id: str) -> str:
#     """
#     Removes a ticket and its associated child records using raw SQL.
#     """
#     # Since SQLite might not have PRAGMA foreign_keys = ON by default,
#     # we manually delete from child tables to be safe.
#     queries = [
#         text("DELETE FROM ticket_messages WHERE ticket_id = :tid"),
#         text("DELETE FROM ticket_metadata WHERE ticket_id = :tid"),
#         text("DELETE FROM tickets WHERE ticket_id = :tid")
#     ]

#     try:
#         with engine.begin() as conn:
#             for q in queries:
#                 conn.execute(q, {"tid": ticket_id})
#         return f"Success: Ticket {ticket_id} and associated data deleted."
#     except Exception as e:
#         return f"Error: {str(e)}"

if __name__ == "__main__":
    mcp.run()