# -*- coding: utf-8 -*-
"""
Created on Fri May 30 20:11:19 2025

@author: giuse
"""

# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager, config_list_from_json
import asyncio
import uvicorn

app = FastAPI()

# CORS for testing from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Agent Setup ===
config_list = config_list_from_json("OAI_CONFIG_LIST")

user_proxy = UserProxyAgent("user_proxy", code_execution_config={"use_docker": False})
assistant = AssistantAgent("assistant", llm_config={"config_list": [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY<")}]}

chat_group = GroupChat(
    agents=[user_proxy, assistant],
    messages=[],
    max_round=10,
)
chat_manager = GroupChatManager(groupchat=chat_group, llm_config={"config_list": [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY<")}]}
# === API Schema ===
class ChatRequest(BaseModel):
    user_message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.user_message

    # Inject user message into the proxy
    user_proxy.initiate_chat(chat_manager, message=user_message)

    # Run async group chat
    result = await chat_manager.a_run()

    # Return only the latest assistant response
    assistant_messages = [m["content"] for m in chat_group.messages if m["name"] == "assistant"]
    last_response = assistant_messages[-1] if assistant_messages else "No reply."

    return {"response": last_response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
