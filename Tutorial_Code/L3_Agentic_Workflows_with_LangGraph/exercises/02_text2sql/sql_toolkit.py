from typing import List
import sqlalchemy
from sqlalchemy.engine.base import Engine
from sqlalchemy import text
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig

@tool
def list_tables_tool(config: RunnableConfig) -> List[str]:
    """
    List all tables in database
    """
    db_engine:Engine = config.get("configurable", {}).get("db_engine")
    inspector = sqlalchemy.inspect(db_engine)

    return inspector.get_table_names()


@tool
def get_table_schema_tool(table_name:str, config: RunnableConfig) -> List[str]:
    """
    Get schema information about a table. Returns a list of dictionaries.
    - name is the column name
    - type is the column type
    - nullable is whether the column is nullable or not
    - default is the default value of the column
    - primary_key is whether the column is a primary key or not

    Args:
        table_name (str): Table name
    """
    db_engine:Engine = config.get("configurable", {}).get("db_engine")
    inspector = sqlalchemy.inspect(db_engine)

    return inspector.get_columns(table_name)


@tool
def execute_sql_tool(query:str, config: RunnableConfig) -> int:
    """
    Execute SQL query and return result. 
    This will automatically connect to the database and execute the query.
    However, if the query is not valid, an error will be raised

    Args:
        query (str): SQL query
    """
    db_engine:Engine = config.get("configurable", {}).get("db_engine")
    with db_engine.begin() as connection:
        answer = connection.execute(text(query)).fetchall()

    return answer
