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
"""
# Ensure the OpenAI API key is set in the environment
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

"""
################################ Create OpenAI model client   #############################
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")
    
model_client_openai = OpenAIChatCompletionClient(
    model="gpt-4o-2024-08-06",
    api_key=OPENAI_API_KEY # Optional if you have an OPENAI_API_KEY env variable set.
)

model_client_gemini = OpenAIChatCompletionClient(
    model="gemini-1.5-flash",
    api_key=GOOGLE_API_KEY,
)

#####################################################################################################
global user_proxy, team, loaded_team_state, agents, agent_list, model_client_openai, model_client_gemini
image_url = None
#########################################################################################################
################################## Initialize variables   ##################################################
CONFIG_FILE = "agent_config.json"
agents = {}
agent_list = []
team = None
loaded_team_state = None  # Will hold config if loaded before team is created
task1 =""
print("✅ Environment cleared.")




############################ TEXT TO SPEECH  #########################################
# Store already processed messages to prevent duplicates
processed_messages = set()
stop_execution = False  # Flag to stop when "APPROVE" is reached

import asyncio
from IPython.display import Audio, display
from pydub import AudioSegment
import os
from openai import OpenAI

speech_queue = asyncio.Queue()

async def speak_worker():
    ###Processes speech requests sequentially from the queue.
    global stop_execution
    AGENT_VOICES = {
    "moderator_agent": "onyx",
    "expert_1_agent": "nova",
    "expert_2_agent": "shimmer",
    "hilarious_agent": "alloy",
    "image_agent": "alloy",
    "describe_agent": "nova",
    "creative_agent": "onyx",
    "user": "alloy"
}
    print("PIPPO10")
    while True:
        item = await speech_queue.get()  # Wait for a new message

        agent_name, content = item  # ✅ Unpack tuple into content and source
        if item == ("system", "TERMINATE"):
            print("🛑 Received TERMINATE. Exiting speak_worker...")
            speech_queue.task_done()
            stop_execution = True
            break
        print ("ITEM", item)
        if content.strip():
            print("CONTENT CONTENT",content)
            #print(f"DEBUG - Adding message to queue: {item}")
            processed_messages.add(item)

            # Determine the voice based on the agent
            voice = AGENT_VOICES.get(agent_name, "onyx")  # Default to "onyx" if not found
            filename = "temp_audio.mp3"
            client1 = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            #text_for_audio =content.rsplit("XYZ", 1)[0].strip()
            #print("TEXT FOR AUDIO", text_for_audio)

            # Step 1: Remove trailing "XYZ"
            text_for_audio = content.rsplit("XYZ", 1)[0].strip()

            # Step 2: Replace Markdown links like [text](https://...)
            text_for_audio = re.sub(
                r'\[.*?\]\(https?://\S+\)',
                'You can find the image at the link',
                text_for_audio
            )

            # Step 3: Replace raw URLs like https://...
            text_for_audio = re.sub(
                r'https?://\S+',
                'You can find the image at the link',
                text_for_audio
            )
            """
            # Step 4: Replace base64 image URIs (optional)
            text_for_audio = re.sub(
                r'data:image/\w+;base64,\S+',
                'You can find the image at the link',
                text_for_audio
            )

            # Clean up extra spaces
            text_for_audio = re.sub(r'\s+', ' ', text_for_audio).strip()

            """
            response = client1.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text_for_audio
            )
            #input=content.strip()
            with open(filename, 'wb') as file:
                file.write(response.content)
            print(f"File saved: {filename}")

            # Display and play audio
            display(Audio(filename, autoplay=True))

            # Load and play audio
            audio_l = AudioSegment.from_mp3(filename)
            audio_duration_seconds = len(audio_l) / 1000

            await asyncio.sleep(audio_duration_seconds + 1)  # ✅ Non-blocking sleep
            os.remove(filename)

        print("Finished speaking.")
        speech_queue.task_done()



# Initialize the assistant agent with the LLM configuration
assistant = AssistantAgent(
            name="assistant",
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
"""
# Define the /chat POST endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    try:
        # Use the assistant agent to process the user's message
        result = await assistant.run(task=user_message)
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
"""
import traceback


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            print(f"📩 Received from browser: {data}")

            if data == "__ping__":
                continue

            try:
                result = await assistant.run(task=data)
                reply_text = ""
                if result and hasattr(result, "messages") and result.messages:
                    last_msg = result.messages[-1]
                    reply_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

                print(f"📤 Sent to browser: {reply_text}")
                await websocket.send_text(reply_text)

            except Exception as inner_error:
                print("❌ Error during assistant response:")
                traceback.print_exc()
                await websocket.send_text("⚠️ Internal error")

    except WebSocketDisconnect:
        print("❌ WebSocket disconnected")