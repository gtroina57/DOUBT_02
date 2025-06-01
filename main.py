# -*- coding: utf-8 -*-
"""
Created on Fri May 30 20:11:19 2025

@author: giuse
"""

# main.py
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

try:
    from autogen import AssistantAgent
except ImportError:
    # If direct import fails, try the older agentchat module path
    from autogen.agentchat import AssistantAgent

# Ensure the OpenAI API key is set in the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

# Configure the LLM for the assistant agent (e.g., using GPT-4 with the API key)
llm_config = {
    "config_list": [
        {"model": "gpt-4", "api_key": OPENAI_API_KEY}
    ]
}

# Initialize the assistant agent with the LLM configuration
assistant_agent = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="You are a helpful AI assistant."
)

# Create the FastAPI app
app = FastAPI()

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