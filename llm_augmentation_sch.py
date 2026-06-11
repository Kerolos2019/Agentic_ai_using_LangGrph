from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import logging

load_dotenv()


# ── Existing tools ─────────────────────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city"""
    weather_data = {
        "New York": "Sunny, 72 F",
        "London": "Cloudy, 15 F",
        "Tokyo": "Rainy, 20 C"
    }
    return weather_data.get(city, "Weather data not available")


@tool
def calculate_tip(bill_amount: float, tip_percentage: float) -> float:
    """Calculate tip amount based on bill and percentage"""
    return round(bill_amount * (tip_percentage / 100), 2)


# ── Tavily search tool 

tavily_search = TavilySearch(
    max_results=3,
    search_depth="basic",
)


# ── Bind all tools to LLM 

llm = ChatOpenAI(model="gpt-4o")

llm_with_tools = llm.bind_tools([
    get_weather,
    calculate_tip,
    tavily_search,         
])


# ── Test prompts 

weather_prompt = "What's the weather in London"
tip_prompt     = "Calculate a 20% tip on a $50 bill"
search_prompt  = "Search for the latest news about ai agents"   # ← triggers Tavily


response = llm_with_tools.invoke(tip_prompt)

tool_calls = response.tool_calls
print("Tool calls:", tool_calls)

#print("Response:", response.content)

# ── Execute whichever tool the LLM chose

for tool_call in tool_calls:
    tool_name = tool_call['name']

    if tool_name == "get_weather":
        result = get_weather.invoke(tool_call['args'])

    elif tool_name == "calculate_tip":
        result = calculate_tip.invoke(tool_call['args'])

    elif tool_name == "tavily_search":
         result = tavily_search.invoke(tool_call['args'])

    else:
        result = "No tool found"

    print(f"\n[{tool_name}] result:\n{result}")