# -*- coding: utf-8 -*-
"""
Created on Fri May 30 20:11:19 2025

@author: giuse
"""

# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- Initialize app ---
app = FastAPI()

# --- Allow all origins (CORS) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Later, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define request body schema ---
class ChatInput(BaseModel):
    user_message: str

# --- Temporary placeholder for multi-agent logic ---
def run_multiagent_response(user_message: str) -> str:
    # 🔁 Replace this with call to your AutoGen-based system
    return f"[Mocked response] You said: {user_message}"

# --- POST endpoint ---
@app.post("/chat")
async def chat_endpoint(input_data: ChatInput):
    user_msg = input_data.user_message
    response = run_multiagent_response(user_msg)
    return {"reply": response}

# --- Optional root ---
@app.get("/")
def root():
    return {"status": "Multi-agent AI API running"}