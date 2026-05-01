import datetime
import io
import json
import os
from typing import Callable

import polars as pl
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = "gemini-3.1-flash-lite-preview"


def make_llm_request(prompt: str) -> str:
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    messages = [
        {"role": "developer", "content": "You are a weather assistant."},
        {"role": "user", "content": prompt},
    ]

    tool_definitions, tool_name_to_func = get_tool_definitions()

    # guard: loop limit, we break as soon as we get an answer
    for _ in range(10):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tool_definitions,  # always pass all tools in this example
            tool_choice="auto",
            max_completion_tokens=1000,
            # extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        resp_message = response.choices[0].message
        messages.append(resp_message.model_dump())

        print(f"Generated message: {resp_message.model_dump()}")
        print()

        # parse possible tool calls (assume only "function" tools)
        if resp_message.tool_calls:
            for tool_call in resp_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                # call tool, serialize result, append to messages
                func = tool_name_to_func[func_name]
                func_result = func(**func_args)

                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(func_result),
                        "tool_call_id": tool_call.id,
                    }
                )
        else:
            # no tool calls, we're done
            return resp_message.content

    # we should not get here
    last_response = resp_message.content
    return f"Could not resolve request, last response: {last_response}"


def get_tool_definitions() -> tuple[list[dict], dict[str, Callable]]:
    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "read_remote_csv",
                "description": "Download a CSV file from a URL and return its contents as text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the CSV file to read.",
                        },
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_remote_parquet",
                "description": "Download a Parquet file from a URL and return its contents as text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the Parquet file to read.",
                        },
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_date",
                "description": 'Get current date in the format "Year-Month-Day" (YYYY-MM-DD).',
                "parameters": {},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather_forecast",
                "description": "Get weather forecast at given country, city, and date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "country": {
                            "type": "string",
                            "description": "The country the city is in.",
                        },
                        "city": {
                            "type": "string",
                            "description": "The city to get the weather for.",
                        },
                        "date": {
                            "type": "string",
                            "description": (
                                "The date to get the weather for, "
                                'in the format "Year-Month-Day" (YYYY-MM-DD). '
                                "At most 4 weeks into the future."
                            ),
                        },
                    },
                    "required": ["country", "city", "date"],
                },
            },
        },
    ]

    tool_name_to_callable = {
        "read_remote_csv": read_remote_csv_tool,
        "read_remote_parquet": read_remote_parquet_tool,
        "get_current_date": current_date_tool,
        "get_weather_forecast": weather_forecast_tool,
    }

    return tool_definitions, tool_name_to_callable


def read_remote_csv_tool(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pl.read_csv(io.BytesIO(response.content))
    return df.write_csv()


def read_remote_parquet_tool(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pl.read_parquet(io.BytesIO(response.content))
    return df.write_csv()


def current_date_tool() -> str:
    return datetime.date.today().isoformat()


def weather_forecast_tool(country: str, city: str, date: str) -> str:
    if country.lower() in {"united kingdom", "uk", "england"}:
        return "Fog and rain"
    else:
        return "Sunshine"


if __name__ == "__main__":
    prompt = "What will be weather in Birmingham in two weeks?"
    response = make_llm_request(prompt)
    print("Response:\n", response)

    print()

    prompt = "What will be weather in Warsaw the day after tomorrow?"
    response = make_llm_request(prompt)
    print("Response:\n", response)

    print()

    prompt = "What will be weather in New York in two months?"
    response = make_llm_request(prompt)
    print("Response:\n", response)
