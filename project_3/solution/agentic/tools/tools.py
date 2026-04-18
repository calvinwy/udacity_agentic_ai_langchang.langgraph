import os
from agentic.state_model import *
from sqlalchemy import create_engine, text

# --- Configuration ---
UDAHUB_DB_PATH = os.path.join(os.getcwd(), os.getenv("UDAHUB_DB_PATH", "data/core/udahub.db"))
DB_URL = f"sqlite:///{UDAHUB_DB_PATH}"
engine = create_engine(DB_URL)

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

def get_highest_urgency_ticket() -> HighestUrgencyTicketResult:
    """
    Fetches the ticket with the highest urgency score using raw SQL.
    """
    print(DB_URL)

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