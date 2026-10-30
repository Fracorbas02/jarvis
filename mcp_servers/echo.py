"""MCP server d'echo — uniquement pour tester le harnais sans LLM réel ni outils réseau."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo")


@mcp.tool()
def ping(target: str) -> str:
    """Outil d'echo : répond pong vers la cible reçue."""
    return f"pong vers {target}"


if __name__ == "__main__":
    mcp.run()  # stdio
