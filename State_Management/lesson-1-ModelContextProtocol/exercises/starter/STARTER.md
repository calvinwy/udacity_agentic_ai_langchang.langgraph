# MCP Exercise: Multi-Server Agent (Starter)

In this exercise, you'll build a LangGraph agent that integrates multiple MCP (Model Context Protocol) servers to fetch GitHub repository issues and save summaries to files.

## Understanding the MCP Servers

This exercise includes two pre-built MCP servers:

- **`github_server.py`**: Provides tools to interact with GitHub's REST API, including `get_last_issue()` and `get_issue()` for fetching repository issues
- **`filesystem_server.py`**: Offers secure file operations within a restricted directory, including `write_file()`, `append_to_file()`, and `read_file()`

## Your Task: Configure MCP Server Connections

Your primary task is to configure the MCP server connections in the `setup_graph()` function. The LangGraph implementation is already provided.

Follow the TODOs in the notebook to complete the MCP configuration. Be sure to:

- Configure the `connections` dictionary to properly connect to both MCP servers
- Set up connections for the filesystem server using `./filesystem_server.py`
- Set up connections for the GitHub server using `./github_server.py`
- Use appropriate command, arguments, and transport settings for both servers

## Expected Behavior

When properly configured, your agent should:
1. Use GitHub server tools to fetch the latest issue from the langchain-ai/langgraph repository
2. Process and summarize the issue content
3. Use filesystem server tools to save the summary to `issue_summary.txt`
4. Confirm successful completion of the task

Test your implementation by running the provided example that demonstrates the multi-server integration in action.