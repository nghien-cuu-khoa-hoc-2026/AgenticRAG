import chainlit as cl 
from typing import TypedDict, List
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama
import os
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import asyncio
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from langgraph_agent import agent

#Authentication
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if username == "admin" and password == "admin":
        return cl.User(identifier="admin", metadata={"role" : "admin"})
    
    return None

# Data layer
@cl.data_layer
def get_data_layer():
    conninfo = os.getenv("DATABASE_URL")

    if not conninfo:
        print("DATABSE_URL not found in environment variables.")
        return None
    
    try:
        data_layer = SQLAlchemyDataLayer(conninfo)
        return data_layer
    except Exception as e:
        print(f"Failed to initialize SQLAlchemyDataLayer: {e}")
        return None
    
#Resume chat with proper message loading
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    try:
        steps = thread.get("steps", [])
        messages = []
        for step in steps:
            step_type = step.get("type")
            content = (step.get("output") or "").strip()
            if not content:
                continue

            if step_type == "user_message":
                messages.append(HumanMessage(content=content))
            elif step_type == "assistant_message":
                messages.append(AIMessage(content=content))
        cl.user_session.set("state", {"messages": messages})

    except Exception as e:
        print(f"\nError resuming chat: {e}\n")
        cl.user_session.set("state", {"messages": []})




#Chat start
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("state", {"messages": []})
    # await cl.Message(content="Hello! I'm your AI assistant powered by LangGraph and Ollama. How can I help you today?").send()

@cl.on_message
async def on_message(msg : cl.Message):

    state = cl.user_session.get("state")
    state["messages"].append(HumanMessage(content=msg.content))

    final_state = await agent.ainvoke(state)

    cl.user_session.set("state", final_state)

#Chat stop
@cl.on_stop
def on_stop():
    print("The user stopped the execution !")

#Chat end
@cl.on_chat_end
def on_chat_end():
    print("The user disconnected !")

import chainlit as cl

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Morning routine ideation",
            message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
            icon="/public/idea.png",
        ),

        cl.Starter(
            label="Explain superconductors",
            message="Explain superconductors like I'm five years old.",
            icon="/public/learn.png",
        ),
        cl.Starter(
            label="Python script for daily email reports",
            message="Write a script to automate sending daily email reports in Python, and walk me through how I would set it up.",
            icon="/public/terminal.png",
            command="code",
        ),
        cl.Starter(
            label="Text inviting friend to wedding",
            message="Write a text asking a friend to be my plus-one at a wedding next month. I want to keep it super short and casual, and offer an out.",
            icon="/public/write.png",
        )
    ]
# ...