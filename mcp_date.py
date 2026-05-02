from datetime import datetime

from fastmcp import FastMCP

mcp = FastMCP("Current date")


@mcp.tool(description="Get the current date in YYYY-MM-DD format.")
async def get_current_date() -> str:

    return datetime.now().strftime("%Y-%m-%d")


@mcp.tool(
    description="Get the current date and time in ISO 8601 format (to the second)."
)
async def get_current_datetime() -> str:

    return datetime.now().replace(microsecond=0).isoformat()


if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8002)
