import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAIEmbeddings
from tavily import TavilyClient

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")    # For ChormaDB embeddings
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")    # For Tavily web search tool
# llm_base_url = "https://openai.vocareum.com/v1"
llm_base_url = "https://api.openai.com/v1"

# 1. Initialize the MCP Server
mcp = FastMCP("Cultpass Help Server")

# --- DATABASE SETUP (Simplified for the server context) ---
# Ensure these variables point to your existing persistence directory
chromadb_directory = "vectorstore"
collection_name = "knowledge_vecotr_store"
embeddings_fn = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # base_url=llm_base_url,
    api_key=OPENAI_API_KEY,
)
# Initialize the vector store connection
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings_fn,
    persist_directory=chromadb_directory,
)

# 2. Define the RAG Search Tool
@mcp.tool()
def article_rag_search(query: str, n_results: int = 3) -> str:
    """
    Search the internal Chroma DB for cultpass help articles and documentation.
    Use this for internal policy and specific cultpass product questions.
    """
    results = vector_store.similarity_search(query, k=n_results)
    
    if not results:
        return "No relevant internal articles found."
    
    formatted_results = []
    for i, doc in enumerate(results):
        formatted_results.append(f"--- Article {i+1} ---\n{doc.page_content}")
        
    return "\n\n".join(formatted_results)

# 3. Define the Web Search Tool
@mcp.tool()
def web_search(query: str) -> dict:
    """
    Search the web using Tavily API
    args:
        query (str): Search query
        search_depth (str): Type of search - 'basic' or 'advanced' (default: advanced)
    """
    client = TavilyClient(api_key=TAVILY_API_KEY)
    
    # Perform the search
    search_result = client.search(
        query=query,
        search_depth=search_depth,
        include_answer=True,
        include_raw_content=False,
        include_images=False
    )
    
    # Format the results
    formatted_results = {
        "answer": search_result.get("answer", "No direct answer found."),
        "results": [
            {"title": r["title"], "url": r["url"], "content": r["content"]} 
            for r in search_result.get("results", [])
        ],
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "query": query
        }
    }
    
    return formatted_results


if __name__ == "__main__":
    mcp.run()



    client = TavilyClient(api_key=TAVILY_API_KEY)