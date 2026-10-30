"""MCP server d'echo — uniquement pour tester le harnais sans LLM réel ni outils réseau."""

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

mcp = Server("echo")


@mcp.tool()
def ping(target: str) -> str:
    """Outil d'echo : répond pong vers la cible reçue."""
    return f"pong vers {target}"


if __name__ == "__main__":
    mcp.run()  # stdio
