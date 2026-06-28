import base64
import io
from typing import Annotated, List, Optional

import matplotlib.pyplot as plt
from fastmcp import FastMCP

mcp = FastMCP("Visualization Server")


@mcp.tool(description="Create a line plot from data and return as base64 image.")
def line_plot(
    data: Annotated[
        List[List[float]],
        "List of lists containing numerical data for lines (y values)",
    ],
    title: Annotated[Optional[str], "Title of the plot"] = "Line Plot",
    x_label: Annotated[Optional[str], "Label for the X axis"] = "X",
    y_label: Annotated[Optional[str], "Label for the Y axis"] = "Y",
    legend: Annotated[bool, "Whether to show the legend"] = False,
    x_data: Annotated[
        Optional[List[float]], "Optional x axis values shared across all lines"
    ] = None,
) -> str:
    plt.figure(figsize=(10, 6))

    for i, line in enumerate(data):
        if x_data is not None:
            plt.plot(x_data, line, label=f"Series {i + 1}")
        else:
            plt.plot(line, label=f"Series {i + 1}")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if legend:
        plt.legend()

    plt.grid(True, linestyle="--", alpha=0.7)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return image_base64


if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8003)
