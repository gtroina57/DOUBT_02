# -*- coding: utf-8 -*-
"""
Created on Fri May 30 20:11:19 2025

@author: giuse
"""

# main.py
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination, HandoffTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_core.tools import FunctionTool
from typing import Sequence
from typing import Any, Dict, List


import playwright
from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import OpenAI
from pathlib import Path
from IPython.display import Audio, display, HTML
from pydub import AudioSegment
from pydub.playback import play  # ✅ Import play() to actually play audio
import time
import pprint

import gc

import json
import re

# Ensure the OpenAI API key is set in the environment
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")


model_client_openai = OpenAIChatCompletionClient(
    model="gpt-4o-2024-08-06",
    api_key=OPENAI_API_KEY # Optional if you have an OPENAI_API_KEY env variable set.
)
# Configure the LLM for the assistant agent (e.g., using GPT-4 with the API key)
"""
llm_config = {
    "config_list": [
        {"model": "gpt-4", "api_key": OPENAI_API_KEY}
    ]
}
"""
# Initialize the assistant agent with the LLM configuration
assistant_agent = AssistantAgent(
            name="Assy",
            description="you are a helpful assistant",
            system_message="you are a helpful assistant",
            model_client=model_client_openai,
)

# Create the FastAPI app
app = FastAPI()

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok", "message": "Service is running"}

# Define request and response data models for the /chat endpoint
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# Define the /chat POST endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    try:
        # Use the assistant agent to process the user's message
        result = await assistant_agent.run(task=user_message)
    except Exception as e:
        # Handle errors (e.g., API issues) by returning an HTTP 500 response
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")
    # Extract the assistant's reply from the agent's result
    reply_text = ""
    if result and hasattr(result, "messages") and result.messages:
        # The last message in the result should be the assistant's response
        last_msg = result.messages[-1]
        reply_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    return ChatResponse(reply=reply_text)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        user = UserProxyAgent("user", name="Pippo")
        while True:
            data = await websocket.receive_text()
            print(f"✅ Received: {data}")

            result = await assistant.a_run(
                input={"name": "user", "content": data},
                sender=user
            )

            reply = result.get("content", "⚠️ No content returned")
            await websocket.send_text(reply)

    except WebSocketDisconnect:
        print("❌ WebSocket disconnected")
    except Exception as e:
        print("❌ WebSocket failed:", e)
        traceback.print_exc()