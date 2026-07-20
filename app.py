import datetime
import json
import re

import gradio as gr
import pytz
import requests

OLLAMA_URL = "http://host.docker.internal:11434/api/chat"
MODEL_ID = "qwen2:7b"
MAX_STEPS = 6


def get_weather(location: str, unit: str = "celsius", humidity: bool = False) -> str:
    """Fetches the current weather for a specified location using Open-Meteo (no API key required)."""
    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1},
        timeout=10,
    )
    geo_results = geo_response.json().get("results")
    if not geo_results:
        return f"Could not find a location matching '{location}'."

    place = geo_results[0]
    current_vars = ["temperature_2m", "wind_speed_10m"]
    if humidity:
        current_vars.append("relative_humidity_2m")

    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": ",".join(current_vars),
            "temperature_unit": unit,
        },
        timeout=10,
    )
    current = weather_response.json().get("current")
    if not current:
        return f"Weather data is unavailable for {place['name']}."

    unit_symbol = "F" if unit == "fahrenheit" else "C"
    result = (
        f"The current weather in {place['name']} is {current['temperature_2m']}°{unit_symbol} "
        f"with wind speed {current['wind_speed_10m']} km/h."
    )
    if humidity and "relative_humidity_2m" in current:
        result += f" Humidity is {current['relative_humidity_2m']}%."
    return result


def get_current_time_in_timezone(timezone: str) -> str:
    """Fetches the current local time in a specified timezone (e.g. 'America/New_York')."""
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


TOOLS = {
    "get_weather": get_weather,
    "get_current_time_in_timezone": get_current_time_in_timezone,
}

SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}} if temperature unit is not specified, defaults to "celsius". args can also include {"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}} to specify the temperature unit explicitly.
if in degrees F passed in query convert that to "fahrenheit". Similarly, if in degrees C passed in query convert that to "celsius".
Can also request humidity by including {"humidity": {"type": "boolean"}} in the args.
get_current_time_in_timezone: Get the current local time in a given timezone, args: {"timezone": {"type": "string"}}

The way you use a tool is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

Example use:
{
  "action": "get_weather",
  "action_input": {"location": "New York", "unit": "celsius", "humidity": true}
}

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:
$JSON_BLOB
Observation: the result of the action. This Observation is unique, complete, and the source of truth.
... (this Thought/Action/Observation can repeat N times)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer, and to only use a tool when you actually need one.
"""


def call_ollama(messages: list[dict]) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_ID,
            "messages": messages,
            "stream": False,
            # Stop right after the model emits an action, before it can hallucinate
            # its own Observation/Final Answer instead of waiting for the real tool result.
            "options": {"temperature": 0.5, "stop": ["Observation:"]},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def run_agent(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]

    for _ in range(MAX_STEPS):
        content = call_ollama(messages)
        messages.append({"role": "assistant", "content": content})

        action_match = re.search(r"\{.*\}", content, re.DOTALL)

        if action_match is None and "Final Answer:" in content:
            return content.split("Final Answer:", 1)[1].strip()

        if not action_match:
            messages.append(
                {
                    "role": "user",
                    "content": "Observation: No action JSON found. Respond with either an Action JSON blob or a Final Answer.",
                }
            )
            continue

        try:
            action = json.loads(action_match.group(0))
            tool_name = action["action"]
            tool_input = action.get("action_input", {})
        except (json.JSONDecodeError, KeyError) as e:
            messages.append({"role": "user", "content": f"Observation: Could not parse action ({e})."})
            continue

        tool_fn = TOOLS.get(tool_name)
        if tool_fn is None:
            observation = f"Unknown tool '{tool_name}'. Available tools: {', '.join(TOOLS)}."
        else:
            try:
                observation = tool_fn(**tool_input) if isinstance(tool_input, dict) else tool_fn(tool_input)
            except Exception as e:
                observation = f"Tool '{tool_name}' raised an error: {e}"

        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "I couldn't reach a final answer within the step limit."


def chat_fn(message: str, history: list[dict]) -> str:
    return run_agent(message)


demo = gr.ChatInterface(
    chat_fn,
    title="Weather Agent",
    type="messages",
)

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
