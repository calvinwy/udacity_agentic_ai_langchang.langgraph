#!/usr/bin/env python
"""
Simple file system MCP server implemented with the `mcp` library.

This server exposes a few safe file operations as MCP tools.  The
`write_file`, `append_to_file`, and `read_file` tools operate within a
specified base directory (by default the current working directory).  To
protect the host system, all file paths are resolved relative to the base
directory and validated to ensure they stay within it.  If you need to
restrict the server further, pass a path on the command line when starting
the server to limit access.
"""

import os
import sys
from mcp.server.fastmcp import FastMCP

# Determine the base directory for all file operations.  If the user
# specifies a directory on the command line, use that; otherwise use
# the current working directory.  Resolving to an absolute path helps
# prevent directory traversal attacks when checking whether a requested
# path is within this base.
BASE_DIR = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.getcwd()

mcp = FastMCP("filesystem")


def _resolve_path(relative_path: str) -> str:
    """Resolve a user‑provided path relative to the base directory.

    Args:
        relative_path: A path relative to the base directory.

    Returns:
        The absolute path corresponding to the provided relative path.

    Raises:
        ValueError: If the resolved path escapes the base directory.
    """
    normalized = os.path.normpath(os.path.join(BASE_DIR, relative_path))
    abs_base = os.path.abspath(BASE_DIR)
    abs_target = os.path.abspath(normalized)
    if not abs_target.startswith(abs_base):
        raise ValueError("Invalid path: outside of allowed base directory")
    return abs_target


@mcp.tool()
def write_file(relative_path: str, content: str) -> str:
    """Write text to a file within the allowed directory.

    If the file does not exist it will be created.  Existing files
    will be overwritten.  Intermediate directories are created as
    necessary.

    Args:
        relative_path: Path of the file relative to the base directory.
        content: Text content to write to the file.

    Returns:
        A confirmation message with the number of characters written.
    """
    path = _resolve_path(relative_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} characters to {relative_path}"


@mcp.tool()
def append_to_file(relative_path: str, content: str) -> str:
    """Append text to a file within the allowed directory.

    If the file does not exist it will be created.  Intermediate
    directories are created as necessary.

    Args:
        relative_path: Path of the file relative to the base directory.
        content: Text content to append to the file.

    Returns:
        A confirmation message with the number of characters appended.
    """
    path = _resolve_path(relative_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)
    return f"Appended {len(content)} characters to {relative_path}"


@mcp.tool()
def read_file(relative_path: str) -> str:
    """Read the contents of a file within the allowed directory.

    Args:
        relative_path: Path of the file relative to the base directory.

    Returns:
        The file's contents as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = _resolve_path(relative_path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    # Start the server using stdio transport.  LangGraph clients spawn
    # this process and communicate with it over standard input/output.
    mcp.run(transport="stdio")