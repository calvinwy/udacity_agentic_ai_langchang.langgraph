# MCP Exercise Solution

## Overview

This solution demonstrates how to configure multiple MCP (Model Context Protocol) servers in a LangGraph agent. The exercise focuses on connecting filesystem and GitHub servers to create an agent that can fetch repository issues and save summaries.

## Key Solution Components

### MCP Server Configuration

The core of the solution is the `connections` dictionary that defines how to spawn and communicate with MCP servers:

```python
connections = {
    "filesystem": {
        "command": "python",
        "args": ["./filesystem_server.py"],
        "transport": "stdio",
    },
    "github": {
        "command": "python",
        "args": ["./github_server.py"],
        "transport": "stdio",
    },
}
```

### How It Works

1. **MultiServerMCPClient**: Uses the connections configuration to spawn server processes
2. **Tool Aggregation**: Automatically combines tools from both servers into a unified set
3. **LangGraph Integration**: The agent seamlessly uses tools from either server as needed
4. **Multi-Step Orchestration**: Demonstrates fetching GitHub data and saving to files

## Expected Workflow

When executed, the solution:
1. Connects to both MCP servers simultaneously
2. Uses GitHub server tools to fetch the latest issue from langchain-ai/langgraph
3. Processes and summarizes the issue content
4. Uses filesystem server tools to save the summary to `issue_summary.txt`
5. Confirms successful completion

## Key Learning Points

- **Server Independence**: Each MCP server runs in its own process with defined boundaries
- **Tool Discovery**: The MultiServerMCPClient automatically discovers and aggregates available tools
- **Seamless Integration**: The LangGraph agent doesn't need to know which server provides which tool
- **Scalable Architecture**: Additional servers can be easily added by extending the connections dictionary

This solution showcases the modularity and composability of the MCP architecture for building capable AI agents.