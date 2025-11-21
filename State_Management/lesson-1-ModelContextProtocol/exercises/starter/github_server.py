#!/usr/bin/env python
"""
Simple GitHub MCP server.

This server exposes tools for retrieving information about issues in
public GitHub repositories.  It uses the GitHub REST API to fetch
issue data.  If a `GITHUB_TOKEN` environment variable is defined, it
will be used to authenticate requests and increase the API rate limit.
Without a token the API still works for public repositories but is
subject to stricter rate limiting.

The implementation follows the Model Context Protocol (MCP) pattern
for custom servers: define a `FastMCP` object, decorate functions
with `@mcp.tool()`, and call `mcp.run()` to start the server【933798328899630†L214-L243】.
"""

import os
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("github")


def _github_request(url: str, params: dict | None = None) -> requests.Response:
    """Send an authenticated (if possible) request to the GitHub API.

    A `GITHUB_TOKEN` environment variable may be set to increase the
    rate limit.  The User‑Agent header is required by GitHub.

    Args:
        url: Full URL to query.
        params: Optional query parameters.

    Returns:
        The `requests.Response` object.
    """
    headers = {"User-Agent": "MCP-GitHub-Server"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return requests.get(url, headers=headers, params=params or {})


@mcp.tool()
def get_last_issue(owner: str, repo: str) -> dict:
    """Fetch the most recently created issue from a GitHub repository.

    Args:
        owner: The GitHub organization or username.
        repo: The repository name.

    Returns:
        A dictionary containing basic details about the issue: number,
        title, body, created_at, and html_url.  If the repository has no
        issues, an empty dict is returned.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    params = {
        "state": "all",
        "per_page": 1,
        "sort": "created",
        "direction": "desc",
    }
    response = _github_request(api_url, params=params)
    response.raise_for_status()
    issues = response.json()
    if not issues:
        return {}
    issue = issues[0]
    return {
        "number": issue.get("number"),
        "title": issue.get("title"),
        "body": issue.get("body", ""),
        "created_at": issue.get("created_at"),
        "html_url": issue.get("html_url"),
    }


@mcp.tool()
def get_issue(owner: str, repo: str, number: int) -> dict:
    """Retrieve details for a specific issue by number.

    Args:
        owner: Repository owner or organization.
        repo: Repository name.
        number: Issue number.

    Returns:
        A dictionary with the issue's number, title, body, created_at,
        and html_url.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"
    response = _github_request(api_url)
    response.raise_for_status()
    issue = response.json()
    return {
        "number": issue.get("number"),
        "title": issue.get("title"),
        "body": issue.get("body", ""),
        "created_at": issue.get("created_at"),
        "html_url": issue.get("html_url"),
    }


if __name__ == "__main__":
    # Start serving on stdio.  Clients discover this server as the
    # "github" server and can call the exposed tools.
    mcp.run(transport="stdio")