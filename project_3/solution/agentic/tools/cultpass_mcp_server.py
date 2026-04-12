import os
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP
from sqlalchemy import create_engine, text
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --- Configuration ---
CULTPASS_DB_PATH = os.getenv("CULTPASS_DB_PATH", "data/external/cultpass.db")
DB_URL = f"sqlite:///{CULTPASS_DB_PATH}"
engine = create_engine(DB_URL)

# Initialize FastMCP server
mcp = FastMCP("Cultpass Manager")

@mcp.tool()
def get_user_subscription_details(user_id: str) -> Dict[str, Any]:
    """
    Returns the user's subscription status and tier using direct engine execution.
    """
    sql = text("""
        SELECT u.full_name, u.is_blocked, u.email, s.status, s.tier 
        FROM users u
        JOIN subscriptions s ON u.user_id = s.user_id
        WHERE u.user_id = :user_id
    """)
    
    with engine.connect() as conn:
        result = conn.execute(sql, {"user_id": user_id}).mappings().first()
        
        if not result:
            result = {"messages": [AIMessage(content="User not found")]}
            return result
            # return {"error": "User or subscription not found."}
        else:
            result = dict(result)
            result["messages"] = AIMessage(content="User information successful retrieved")
            result["user_name"] = result.pop("full_name")
            result["user_status"] = result.pop("status")
            result["user_tier"] = result.pop("tier")
            result["user_is_blocked"] = bool(result.pop("is_blocked"))
            result["user_email"] = result.pop("email")
            return result

@mcp.tool()
def get_user_reservation_history(user_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves all reservations for a specific user.
    """
    sql = text("""
        SELECT 
            r.reservation_id, 
            r.status, 
            e.title AS experience_title, 
            e.location, 
            e.when AS date
        FROM reservations r
        JOIN experiences e ON r.experience_id = e.experience_id
        WHERE r.user_id = :user_id
    """)
    
    with engine.connect() as conn:
        results = conn.execute(sql, {"user_id": user_id}).mappings().all()
        
        # SQLite returns dates as strings or datetime objects depending on driver config
        # We ensure they are strings for the MCP output
        history = []
        for row in results:
            item = dict(row)
            if hasattr(item.get("date"), "isoformat"):
                item["date"] = item["date"].isoformat()
            history.append(item)
            
        return history

@mcp.tool()
def get_experience_availability(experience_id: str) -> Dict[str, Any]:
    """
    Checks slots available using direct SQL.
    """
    sql = text("SELECT title, slots_available FROM experiences WHERE experience_id = :id")
    
    with engine.connect() as conn:
        result = conn.execute(sql, {"id": experience_id}).mappings().first()
        return dict(result) if result else {"error": "Not found"}

@mcp.tool()
def unblock_user(user_id: str) -> str:
    """
    Unblock a user's account and commit the transaction immediately.
    """
    # In direct connection mode, we need to handle the transaction commit
    check_sql = text("SELECT full_name FROM users WHERE user_id = :user_id AND is_blocked = 1")
    update_sql = text("UPDATE users SET is_blocked = 0 WHERE user_id = :user_id")
    
    with engine.begin() as conn:  # .begin() automatically commits at the end of the block
        user = conn.execute(check_sql, {"user_id": user_id}).mappings().first()
        
        if not user:
            return f"User {user_id} not found or already unblocked."
        
        conn.execute(update_sql, {"user_id": user_id})
        return f"Success: {user['full_name']} has been unblocked."

if __name__ == "__main__":
    mcp.run()