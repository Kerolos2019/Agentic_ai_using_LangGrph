from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
#from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

import os
load_dotenv()
"""
"example:   DB_URI = "postgresql://postgres:root@localhost:5432/langgraph_stm?sslmode=disable""

"""
DB_URI = DB_URI = os.getenv("DB_URI")


llm = ChatOpenAI(model="gpt-4o")

def chatbot(state: MessagesState):

    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }


builder = StateGraph(MessagesState)

builder.add_node(chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:

    #checkpointer.setup() # Call this only the first time

    graph = builder.compile(checkpointer=checkpointer)


    config = {
        "configurable": {
            "thread_id": "chat_session_1"
        }
    }

    # Turn 1
    message_1 = "Hi! My name is ashraf, I am an AI osta"

    input_1 = {
        "messages": [HumanMessage(content=message_1)]
    }

    result_1 = graph.invoke(input_1, config=config)

    print(f"User: {message_1}")
    print(f"AI: {result_1['messages'][-1].content}")

    # Turn 2
    message_2 = "What's my name?"

    input_2 = {
        "messages": [HumanMessage(content=message_2)]
    }

    result_2 = graph.invoke(input_2, config=config)

    print(f"User: {message_2}")
    print(f"AI: {result_2['messages'][-1].content}")