from typing import Annotated
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from dotenv import load_dotenv

load_dotenv()


#1. Tools 

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city"""
    print('tool wheather is called')
    weather_data = {
        "New York": "Sunny, 72 F",
        "London": "Cloudy, 15 F",
        "Tokyo": "Rainy, 20 C"
    }
    return weather_data.get(city, "Weather data not available")


@tool
def calculate_tip(bill_amount: float, tip_percentage: float) -> float:
    """Calculate tip amount based on bill and percentage"""
    print('tool tip_calc is called')

    return round(bill_amount * (tip_percentage / 100), 2)


tavily_search = TavilySearch(max_results=3, search_depth="basic")

tools = [get_weather, calculate_tip, tavily_search]


#  2. LLM with tools bound

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)


#  3. State definition 

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   


#  4. Nodes 

def call_llm(state: AgentState) -> AgentState:
    """Send messages to the LLM and get a response (possibly with tool calls)."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}          


tool_node = ToolNode(tools)                  # handles ALL tool execution automatically


# 5. Routing logic

def should_use_tools(state: AgentState) -> str:
    """Check if the last message contains tool calls → route accordingly."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ── 6. Build the graph ─────────────────────────────────────────────────────────

graph_builder = StateGraph(AgentState)

graph_builder.add_node("llm",   call_llm)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("llm")

graph_builder.add_conditional_edges(
    "llm",
    should_use_tools,         # decides: "tools" or END
    {"tools": "tools", END: END}
)

graph_builder.add_edge("tools", "llm")   # after tools run → back to LLM for final answer

graph = graph_builder.compile()


# ── 7. Run it ──────────────────────────────────────────────────────────────────

prompts = [
    "What's the weather in London?",
    "Calculate a 20% tip on a $50 bill",
    "Search for the latest news about AI agents",
]
with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

print("Graph saved to graph.png")

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print('='*60)

    result = graph.invoke({"messages": [HumanMessage(content=prompt)]})

    #  LLM's final answer
    final = result["messages"][-1]
    print("Final answer:", final.content)





    