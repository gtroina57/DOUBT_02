# -*- coding: utf-8 -*-
"""
Created on Fri May 30 20:11:19 2025

@author: giuse
"""

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


from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path


# Create the FastAPI app
app = FastAPI()
# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the index.html at root
@app.get("/", response_class=HTMLResponse)
async def get_index():
    html_path = Path("static/index.html")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


#####################################################################################################
#global user_proxy, team, loaded_team_state, agents, agent_list, model_client_openai, model_client_gemini

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
    api_key=GOOGLE_API_KEY
)

#####################################################################################################
image_url = None
#########################################################################################################
################################## Initialize variables   ##################################################
CONFIG_FILE = "agent_config.json"
agents = {}
agent_list = []
team = None
loaded_team_state = None  # Will hold config if loaded before team is created
task1 ="This is a debate on ethics and AI"
print("✅ Environment cleared.")

############################ TEXT TO SPEECH  #########################################
# Store already processed messages to prevent duplicates
processed_messages = set()
stop_execution = False  # Flag to stop when "APPROVE" is reached

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

##########################################################################################################
################################# Build Agents from configuration  ####################################

model_clients_map = {
    "openai": model_client_openai,
    "gemini": model_client_gemini
}
##########################################################################################################
################################# Load default configuration    ##########################################
with open(CONFIG_FILE, "r") as f:
    agent_config = json.load(f)
    
##########################################################################################################
################################# Build name_to_agent_skill for introducing Agents #######################
def extract_agent_skills(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    skills = []
    for agent_id, cfg in config.items():
        description = cfg.get("description", "")
        words = description.split()
        name = words[1] if len(words) > 1 else agent_id  # crude fallback
        skills.append(f"{name} ({description.split(',', 1)[-1].strip()})")

    #skills.append("Giuseppe (user proxy)")
    return ", ".join(skills)

##########################################################################################################
################################# Build Agents from configuration  #######################################

tool_lookup = {
}
##########################################################################################################
################################# Build Agents from configuration  #######################################
def build_agents_from_config(config_path, name_to_agent_skill, model_clients_map):
    global task1
    with open(config_path, "r") as f:
        config = json.load(f)

    agents = {}
    for name, cfg in config.items():
        if name == "proxy_agent":
            continue  # Skip creating an AssistantAgent for the user_proxy
        sys_msg = (
            cfg["system_message"]
            .replace("{task1}", task1)
            .replace("{name_to_agent_skill}", name_to_agent_skill)
        )

        client_key = cfg.get("model_client", "openai")
        model_client = model_clients_map.get(client_key)
        if model_client is None:
            raise ValueError(f"Model client '{client_key}' not found for agent '{name}'")
        print("MODEL_CLIENT", model_client)

        tool_names = cfg.get("tools", [])
        if isinstance(tool_names, str):
            import ast
            tool_names = ast.literal_eval(tool_names)  # Safe way to convert string to list

        tool_list = [tool_lookup[t] for t in tool_names] if tool_names else []

        agent = AssistantAgent(
            name=name,
            description=cfg["description"],
            system_message=sys_msg,
            model_client=model_client,
            tools=tool_list)

        # Ensure llm_config exists
        if not hasattr(agent, "llm_config") or agent.llm_config is None:
            agent.llm_config = {}
        agent.llm_config["temperature"] = cfg.get("temperature", 0.7)

        agents[name] = agent
        print(f"✅ Initialized {len(agents)} agents for debate topic: {task1}")
    return agents


#print(f"✅ Initialized {len(agents)} agents for debate topic: {task1}")
    

##########################################################################################################
################################# Termination  ####################################
text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=80)
termination = text_mention_termination | max_messages_termination

##########################################################################################################
################################# Selector Prompt Function    ############################################
selector_prompt = """
You are the Selector agent following strictly the instructions of the moderator and of the selector_func
"""

def dynamic_selector_func(thread):
    global agent_id
    last_msg = thread[-1]
    last_message = last_msg.content.lower().strip()
    sender = last_msg.source.lower()

    name_to_agent = {
        "alice": "expert_1_agent",
        "bob": "expert_2_agent",
        "charlie": "hilarious_agent",
        "alan": "moderator_agent",
        "albert": "creative_agent",
        "giuseppe": "user_proxy",
    }

    # 🔹 First user interaction → go to moderator
    if sender == "user":
        print("👤 User input detected. Moderator takes over.")
        return "moderator_agent"

    # 🔹 AGENT (not moderator) just spoke
    if sender != "moderator_agent":
        if last_message.endswith("xyz"):
            focus_area = last_message.rsplit("xyz", 1)[0].strip()
            pattern = r'\b(' + '|'.join(map(re.escape, name_to_agent.keys())) + r')\b'
            matches = re.findall(pattern, focus_area)
            unique_mentions = set(matches)

            if len(unique_mentions) == 1:
                mentioned = matches[0]
                if name_to_agent[mentioned] == sender:
                    print(f"🔁 Agent '{sender}' mentioned only themselves. Returning to Moderator.")
                    return "moderator_agent"
                else:
                    print(f"📣 Agent '{sender}' mentioned another agent ('{mentioned}'). Moderator should intervene.")
                    return "moderator_agent"
            elif len(unique_mentions) > 1:
                print(f"📣 Agent '{sender}' mentioned multiple agents. Moderator should intervene.")
                return "moderator_agent"
        # No 'xyz' or no mentions → let agent continue
        print(f"⏭ Agent '{sender}' keeps the floor.")
        return sender

    # 🔹 MODERATOR just spoke
    if not last_message.endswith("xyz"):
        print("⚠️ Moderator message incomplete (no 'xyz'). Staying with Moderator.")
        return "moderator_agent"

    focus_area = last_message.rsplit("xyz", 1)[0].strip()
    pattern = r'\b(' + '|'.join(map(re.escape, name_to_agent.keys())) + r')\b'
    matches = list(re.finditer(pattern, focus_area))

    if not matches:
        print("⚠️ No agent mentioned by moderator. Staying with Moderator.")
        return "moderator_agent"

    # ✅ Find the last valid agent mentioned
    for match in matches:

        if len(matches) == 1:
            match = matches[0]
            name = match.group(1)
            agent_id = name_to_agent.get(name)
            if agent_id and agent_id != "moderator_agent":
                print(f"✅ Moderator selected '{name}'. Routing to {agent_id}.")
                return agent_id

        if len(matches) >= 2:
            match = matches[1]
            name = match.group(1)
            agent_id = name_to_agent.get(name)
            if agent_id and agent_id != "moderator_agent":
                print(f"✅ Moderator selected '{name}'. Routing to {agent_id}.")
                return agent_id
    print("⚠️ Moderator mentioned only user or moderator. Staying with Moderator.")
    return "moderator_agent"

#print(dir(agents["user_proxy"]))
#print(dir (team))
#print(help (team))

####################################################################################################
####################################################################################################
####################################################################################################
# === Globals ===
user_conversation = []
gradio_input_buffer = {"message": None}
agent_config_ui = {}
## chat_log_shared = gr.Textbox(label="Conversation Log", lines=20, interactive=False)

################## SET TOPIC        ###########################################################
def set_task_only(task_text):
            global task1
            task1 = task_text
            return "✅ Debate topic set." if task_text else "❌ Topic cannot be empty."

##########################################################################################################
################################# Configuration File    ###################################################=
def load_agent_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def save_agent_config(*args):
    updated = {}
    idx = 0
    for name in agent_config_ui:
        updated[name] = {
            "description": args[idx],
            "system_message": args[idx + 1],
            "temperature": args[idx + 2],
            "model_client": args[idx + 3],
            "tools": args[idx + 4]
        }
        idx += 5
    with open(CONFIG_FILE, "w") as f:
        json.dump(updated, f, indent=2)
    return "✅ Configuration saved."


##########################################################################################################
################################# SAVE Config    ###################################################
# === Save config ===
async def save_config():
    global team
    team_state = await team.save_state()
    with open("coding/team_state.json", "w") as f:
        json.dump(team_state, f)
    print("✅ Config saved.")
    return "✅ Config saved to disk."

def sync_save_config():
    return asyncio.run(save_config())  # 'team' must be globally accessible


##########################################################################################################
################################# LOAD Config   ###################################################
def sync_load_config():
    global loaded_team_state
    with open("coding/team_state.json", "r") as f:
        loaded_team_state = json.load(f)
    print("🟢 Config loaded and pending application.")
    return "🟢 Config loaded. Ready to apply when system starts."

##########################################################################################################
################################# User Intervention    ###################################################
# === User interaction ===
def handle_user_message(message):
    global user_conversation, gradio_input_buffer
    gradio_input_buffer["message"] = message
    user_conversation.append(f"🧑 USER: {message}")
    return "\n".join(user_conversation)

def intervene_now(user_input):
    return handle_user_message(user_input)


##########################################################################################################
################################# loop for debate  #######################################################
async def run_chat(team, websocket=None):
    global stop_execution, image_url, task1, awaiting_user_reply, gradio_input_buffer

    async for result in team.run_stream(task=task1):
        if stop_execution:
            break

        if hasattr(result, "content") and isinstance(result.content, str):
            text = result.content
            agent_name = result.source

            # ✅ Notify frontend if it's Giuseppe's turn
            if agent_name == "user_proxy" and websocket:
                awaiting_user_reply = True
                await websocket.send_text("__USER_PROXY_TURN__")
                
            # ✅ Optional: log internal message history
            if not hasattr(team, "_message_history"):
                team._message_history = []
            team._message_history.append({"sender": agent_name, "content": text})

            # ✅ Log to console
            print(f"👤 sender: {agent_name}")
            print(f"📝 content: {text}")
            print(f"🖼️ image_url: {image_url}")

            # ✅ Add to chat log (for UI)
            prefix = "🧑" if "user" in agent_name.lower() else "🤖"
            user_conversation.append(f"{prefix} {agent_name.upper()}: {text}")

            # ✅ Push to speech queue
            await speech_queue.put((agent_name, text))

            # ✅ Terminate if "TERMINATE" is found
            if "TERMINATE" in text:
                stop_execution = True
                await speech_queue.put(("system", "TERMINATE"))
                print("✅ Chat terminated.")
                break

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok", "message": "Service is running"}

import traceback

#########

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global team, agents, agent_list, stop_execution, loaded_team_state

    session = {
        "awaiting_user_reply": False,
        "first_user_input": True,
        "early_input_buffer": None,
    }
    user_message_queue = asyncio.Queue()
    async def flush_queue(queue: asyncio.Queue):
        while not queue.empty():
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                break
    await flush_queue(user_message_queue)

    await websocket.accept()
    try:
        if not team:
            # 🔧 Load agents from config
            name_to_agent_skill = extract_agent_skills(CONFIG_FILE)
            agents = build_agents_from_config(CONFIG_FILE, name_to_agent_skill, model_clients_map)

            # ✅ Define the input function using queue
            async def websocket_async_input_func(*args, **kwargs):
                while True:
                    msg = await user_message_queue.get()
                    if msg.strip():  # 🚫 Skip empty input
                        return msg
                    print("⚠️ Skipped empty user input")

            # 👤 Add the user_proxy agent with the queue-based input_func
            agents["user_proxy"] = UserProxyAgent(name="user_proxy", input_func=websocket_async_input_func)

            # 🧠 Define agent list
            agent_list = [
                agents["moderator_agent"],
                agents["expert_1_agent"],
                agents["expert_2_agent"],
                agents["hilarious_agent"],
                agents["creative_agent"],
                agents["user_proxy"],
            ]

            # 🧠 Initialize SelectorGroupChat
            team = SelectorGroupChat(
                agent_list,
                model_client=model_client_openai,
                selector_func=dynamic_selector_func,
                termination_condition=termination,
                allow_repeated_speaker=True,
            )

            if loaded_team_state:
                await team.load_state(loaded_team_state)
                loaded_team_state = None

            # 🚀 Start core tasks
            asyncio.create_task(run_chat(team))      # Debate engine
            asyncio.create_task(speak_worker())      # Audio playback

        # 🔁 Main WebSocket message loop
        while True:
            data = await websocket.receive_text()
            if data == "__ping__":
                continue

            stop_execution = False

            # 🎯 Handle first user input to set debate topic
            if session["first_user_input"]:
                print("📌 First user input received, setting debate topic.")
                await user_message_queue.put(data)
                session["first_user_input"] = False
                continue

            # 🕓 Handle moderator giving floor to user
            if data == "__USER_PROXY_TURN__":
                print("🟢 Moderator has delegated to user_proxy.")
                session["awaiting_user_reply"] = True

                # 🕗 Check if early input was already given
                if session["early_input_buffer"]:
                    print("📥 Using cached early input:", session["early_input_buffer"])
                    await user_message_queue.put(session["early_input_buffer"])
                    session["early_input_buffer"] = None
                    session["awaiting_user_reply"] = False
                else:
                    await websocket.send_text("👤 It's your turn to speak.")
                continue

            # 🎤 User is replying during their turn
            if session["awaiting_user_reply"]:
                print("👤 User replied:", data)
                await user_message_queue.put(data)
                session["awaiting_user_reply"] = False
            else:
                # 🕒 Not their turn: warn and buffer if useful
                if not session["early_input_buffer"]:
                    session["early_input_buffer"] = data
                    print("⚠️ Cached early user input:", data)
                await websocket.send_text("⚠️ Not your turn yet. Please wait for the moderator.")

            await speech_queue.join()

    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
        await speech_queue.put(("system", "TERMINATE"))
        stop_execution = True
        team = None

    except Exception as e:
        traceback.print_exc()
        await websocket.send_text("⚠️ Internal server error")