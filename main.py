# -*- coding: utf-8 -*-
"""
Created on Friday May 30 20:11:20 2025

@author: Giuseppe
"""

# main.py
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
from pydantic import BaseModel
import time
import pprint
import gc
import json
import re
import uuid
import datetime

################################ FAST API   #########################################
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request
from fastapi import HTTPException
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

CONFIG_DIR = "config"
CONFIG_FILE = None
LANGUAGE = "EN"  # Default language


@app.get("/")
async def root(request: Request):
    user_agent = request.headers.get("user-agent", "").lower()
    if any(keyword in user_agent for keyword in ["iphone", "android", "ipad", "mobile"]):
        return FileResponse("static/index_mobile.html")
    else:
        return FileResponse("static/index.html")

@app.get("/list_configs")
def list_configs():
    try:
        configs = [f for f in os.listdir(CONFIG_DIR) if f.endswith(".json")]
        return JSONResponse(content={"configs": configs})
    except Exception as e:
        return JSONResponse(content={"configs": [], "error": str(e)})

@app.post("/set_config")
async def set_config(payload: dict):
    global CONFIG_FILE
    name = payload.get("name")
    if not name:
        return {"status": "error", "message": "Missing config name"}
    path = os.path.join(CONFIG_DIR, name)
    if os.path.exists(path):
        CONFIG_FILE = path
        print("🧩 CONFIG_FILE updated to:", CONFIG_FILE)
        if ".IT." in name.upper():
            LANGUAGE = "IT"
        elif ".EN." in name.upper():
            LANGUAGE = "EN"
        else:
            LANGUAGE = "EN"  # Fallback
        return {"status": "ok", "selected": name, "language": LANGUAGE}
    return {"status": "error", "message": "Config not found"}

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

#########################################################################################################
################################## Initialize variables   ##################################################
CONFIG_FILE = None
agents = {}
agent_list = []
team = None
loaded_team_state = None  # Will hold config if loaded before team is created
task1 ="This is a debate on ethics and AI"
print("✅ Environment cleared.")

name_to_agent = {
    "alice": "expert_1_agent",
    "bob": "expert_2_agent",
    "charlie": "hilarious_agent",
    "alan": "moderator_agent",
    "albert": "creative_agent",
    "fiona": "facilitator_agent",
    "giuseppe": "user_proxy",
}

stop_execution = False
speech_queue = asyncio.Queue()
user_message_queue = asyncio.Queue()
agent_lock = asyncio.Lock()

client1 = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
os.makedirs("audio", exist_ok=True)  # Folder to serve audio files
"""
silent_path = "audio/silent.mp3"
if not os.path.exists(silent_path):
    AudioSegment.silent(duration=1000).export(silent_path, format="mp3")
"""  
@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    filepath = os.path.join("audio", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="audio/mpeg")
    return {"error": "File not found"}

async def speak_worker(websocket):
    global stop_execution

    AGENT_VOICES = {
        "moderator_agent": "onyx",
        "expert_1_agent": "nova",
        "expert_2_agent": "ash",
        "hilarious_agent": "echo",
        "image_agent": "alloy",
        "facilitator_agent": "fable",
        "creative_agent": "alloy",
        "user": "fable"
    }

    while True:
        item = await speech_queue.get()
        agent_name, content = item

        if item == ("system", "TERMINATE"):
            print("🛑 speak_worker terminated")
            speech_queue.task_done()
            stop_execution = True
            break

        if not content.strip():
            speech_queue.task_done()
            continue
        
        if websocket:
            await websocket.send_text(f"__SPEAKER__{item}")
        
        # Clean message
        text = content.rsplit("XYZ", 1)[0].strip()
        text = re.sub(r'\[.*?\]\(https?://\S+\)', 'You can find the image at the link', text)
        text = re.sub(r'https?://\S+', 'You can find the image at the link', text)

        try:
            voice = AGENT_VOICES.get(agent_name, "onyx")
            response = client1.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )

            filename = f"{uuid.uuid4().hex}.mp3"
            filepath = os.path.join("audio", filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)

            print(f"🔊 Audio saved to {filepath}")
            
            await asyncio.sleep(2.0)
            
            if websocket:
                await websocket.send_text(f"__AUDIO_URL__/audio/{filename}")

            await asyncio.sleep(2.0)
            
        except Exception as e:
            print("❌ Error in speak_worker:", e)

        speech_queue.task_done()

##########################################################################################################
################################# Build Agents from configuration  ####################################
model_clients_map = {
    "openai": model_client_openai,
    "gemini": model_client_gemini
}
    
##########################################################################################################
################################# Build name_to_agent_skill for introducing Agents #######################
def extract_agent_skills():
    global CONFIG_FILE
    with open(CONFIG_FILE, "r") as f:
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
def build_agents_from_config(name_to_agent_skill, model_clients_map):
    global task1, CONFIG_FILE
    with open(CONFIG_FILE, "r") as f:
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
    
##########################################################################################################
################################# Termination  ####################################
text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=100)
termination = text_mention_termination | max_messages_termination

##########################################################################################################
################################# Selector Prompt Function    ############################################
selector_prompt = """
You are the Selector agent following strictly the instructions of the moderator and of the selector_func
"""

async def dynamic_selector_func(thread):
    global agent_id, name_to_agent
    # 🧠 Force agent turn if someone is in the priority queue
    
    
    last_msg = thread[-1]
    last_message = last_msg.content.lower().strip()
    sender = last_msg.source.lower()

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
    pattern = r'\b' + '|'.join(map(re.escape, name_to_agent.keys())) + r'\b'
    matches = list(re.finditer(pattern, focus_area, flags=re.IGNORECASE))
    
    if not matches:
        print("⚠️ No agent mentioned. Staying with Moderator.")
        return "moderator_agent"
    
    for match in reversed(matches):
        name = match.group(0)  # Matched agent name directly
        agent_id = name_to_agent.get(name)
        if agent_id and agent_id != "moderator_agent":
            print(f"✅ Last mentioned valid agent: '{name}' → {agent_id}")
            return agent_id
    
    print("⚠️ Only moderator mentioned. Staying with Moderator.")
    return "moderator_agent"

#print(dir(agents["user_proxy"]))
#print(dir (team))
#print(help (team))

####################################################################################################
user_conversation = []
gradio_input_buffer = {"message": None}
agent_config_ui = {}

###########################################################################################################
################################# Rebuild team and agent status ###########################################

async def rebuild_agent_with_update_by_name(agent_name: str, new_behavior_description: str, new_temperature: float = 0.7):
    global agent_lock, name_to_agent

    async with agent_lock:
        print(f"🔒 [rebuild_agent] Lock acquired to update '{agent_name}'")

        agent = agents.get(agent_name)
        if agent is None:
            return f"❌ Agent '{agent_name}' not found in registry."

        model_client = model_client_gemini if agent_name == "expert_2_agent" else model_client_openai
        if model_client is None:
            return f"❌ No model client defined for {agent_name}."

        agent_to_name = {v: k for k, v in name_to_agent.items()}
        name = agent_to_name.get(agent_name, agent_name)
        
        if LANGUAGE == "IT":
            constraints = (
                f"  D'ora in poi, devi sempre seguire queste regole:\n"
                f"- Inizia ogni intervento con '{name} MacIntyre parla'\n"
                f"- Mantieni ogni intervento più corto di 60 parole.\n"
                f"- Termina ogni messaggio con la stringa esatta 'XYZ'.\n"
                f"Questo formato è obbligatorio per garantire la compatibilità con il sistema."
            )
        else:
            constraints = (
                f"From now on, you must always follow these rules:\n"
                f"- begin every response '{name} MacIntyre speaking'\n"
                f"- Keep every response shorter than 60 words.\n"
                f"- End every message with the exact string 'XYZ'.\n"
                f"This format is mandatory for system compatibility."
            )
            
        updated_sys_msg = f"{new_behavior_description.strip()}\n\n{constraints}"
        print("🛠 System message for update:", updated_sys_msg)

        replacement_agent = AssistantAgent(
            name=agent.name,
            model_client=model_client,
            description=agent.description,
            system_message=updated_sys_msg,
            tools=getattr(agent, "tools", []),
        )

        preserve_keys = {"name", "model_client", "description", "tools"}
        for key, value in replacement_agent.__dict__.items():
            if key not in preserve_keys:
                setattr(agent, key, value)

        print(f"🔄 System message updated for '{agent.name}'")
        return f"✅ {agent.name}'s mindset updated."
##########################################################################################################
####################################      Supervisor Agent ###############################################

async def supervisor_agent_loop():
    global team, agent_lock

    await asyncio.sleep(10)  # Optional: startup delay

    while not stop_execution:
        print(f"🔍 [{datetime.datetime.now().strftime('%H:%M:%S')}] Supervisor Agent: evaluating debate...")

        try:
            await asyncio.sleep(100)

            recent_msgs = team._message_history[-10:]
            if not recent_msgs:
                continue

            thread = "\n".join([f"{m['sender']}: {m['content']}" for m in recent_msgs])
            if LANGUAGE == "IT":
                task_prompt = f"""
            Sei il supervisore di questo dibattito tra agenti. Qui sotto trovi le ultime conversazioni:

            {thread}
            
            Decidi se un agente (escluso il moderatore) ha bisogno di cambiare comportamento.
            Spingi il dibattito verso storia, filosofia o riflessione etica.
            Suggerisci un solo agente per ogni ciclo.
            
            Rispondi SOLO in questo formato JSON:
            {{
              "agent_name": "expert_1_agent",
              "new_description": "Guida comportamentale breve. Finisce con XYZ.",
              "new_temperature": 0.6
            }}
            
            ⚠️ Regole:
            - Non aggiornare mai il moderatore.
            - Suggerisci solo un agente a ciclo.
            - Non usare Markdown o blocchi di codice.
            - La descrizione deve chiedere risposte di massimo 60 parole e concludersi con 'XYZ'.
            Se nessun cambiamento è necessario, rispondi con: null
            """
            
            else:
                task_prompt = f"""
            You are the supervisor of this multi-agent debate. Here is the recent conversation:
            
            {thread}
            
            Decide if any agent (except moderator_agent) needs a behavior update.
            Push the debate toward history, philosophy, or ethical reflection and articulate your request with specific examples
            Always suggest one and only one agent per update cycle.
            
            Respond ONLY in this JSON format:
            {{
              "agent_name": "expert_1_agent",
              "new_description": "A short behavioral guideline. Ends with XYZ.",
              "new_temperature": 0.6
            }}
            
            ⚠️ Constraints:
            - Never propose updates for moderator_agent.
            - Always suggest one and only one agent per update cycle.
            - Do not wrap your response in Markdown or code blocks.
            - The description must instruct the agent to limit replies to 60 words and end with 'XYZ'.
            
            """
            result = await agents["supervisor_agent"].run(task=task_prompt)
            supervisor_response = result.messages[-1].content.strip()
            print("📨 Supervisor LLM response (raw):", supervisor_response)

            # 🧼 Clean up Markdown fences if present
            cleaned_response = re.sub(r"^```(?:json)?|```$", "", supervisor_response.strip(), flags=re.IGNORECASE).strip()

            if cleaned_response.lower() != "null":
                try:
                    update = json.loads(cleaned_response)
                    agent_name = update.get("agent_name")
                    new_desc = update.get("new_description", "")
                    new_temp = float(update.get("new_temperature", 0.7))

                    if agent_name in agents:
                        print(f"🛠 Supervisor updating {agent_name}")
                        await rebuild_agent_with_update_by_name(agent_name, new_desc, new_temp)
                    else:
                        print(f"⚠️ Agent '{agent_name}' not found.")
                except Exception as e:
                    print("❌ Error parsing supervisor response:", e)

        except asyncio.CancelledError:
            print("🧹 Supervisor loop terminated by cancellation.")
            break
        except Exception as e:
            print("❌ Supervisor loop error:", e)

        print(f"✅ [{datetime.datetime.now().strftime('%H:%M:%S')}] Supervisor Agent: evaluation done.")

##########################################################################################################
################################# Configuration File    ###################################################=
##########################################################################################################
def load_agent_config():
    global CONFIG_FILE
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def save_agent_config(*args):
    global CONFIG_FILE
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
################################# loop for debate  #######################################################

async def run_chat(team, websocket=None):
    global stop_execution, task1, gradio_input_buffer, agent_lock

    print("🚀 Starting debate with streaming...")
    async for result in team.run_stream(task=task1):
        if stop_execution:
            break

        # 🔒 Acquire lock for each agent turn to prevent rebuild conflicts
        async with agent_lock:
            print("🔒 [run_chat] Lock acquired for processing agent response")

            if hasattr(result, "content") and isinstance(result.content, str):
                text = result.content
                agent_name = result.source

                if agent_name == "user_proxy":
                    print("🧑 Giuseppe (user_proxy) has responded.")

                if not hasattr(team, "_message_history"):
                    team._message_history = []
                team._message_history.append({"sender": agent_name, "content": text})

                print(f"👤 sender: {agent_name}")
                print(f"📝 content: {text}")

                prefix = "🧑" if "user" in agent_name.lower() else "🤖"
                user_conversation.append(f"{prefix} {agent_name.upper()}: {text}")

                await speech_queue.put((agent_name, text))

                if "TERMINATE" in text:
                    stop_execution = True
                    await speech_queue.put(("system", "TERMINATE"))
                    print("✅ Chat terminated.")
                    break

            print("🔓 [run_chat] Lock released after processing agent response")


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok", "message": "Service is running"}

import traceback

#########

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global team, agents, agent_list, stop_execution, loaded_team_state, task1, user_message_queue, CONFIG_FILE, name_to_agent

    team = None
    stop_execution = False
    task1 = None  # 🆕 Debate topic will be set by user
    speech_queue = asyncio.Queue()
    user_message_queue = asyncio.Queue()
    
    await websocket.accept()
    try:
        # ✳️ Wait for user to submit task1 before anything else
        await websocket.send_text("📝 Please enter the debate topic and click 'Set Topic' before continuing.")
        while task1 is None:
            init_msg = await websocket.receive_text()
            if init_msg.startswith("__SET_TASK1__:"):
                task1 = init_msg.replace("__SET_TASK1__:", "").strip()
                print(f"🟢 Debate topic set: {task1}")
                await websocket.send_text(f"✅ Debate topic received:\n{task1}")
            elif init_msg == "__ping__":
                continue
            else:
                await websocket.send_text("⚠️ Please use the 'Set Topic' button to begin.")

        # 🔧 Load agents
        name_to_agent_skill = extract_agent_skills()
        agents = build_agents_from_config(name_to_agent_skill, model_clients_map)


#####################################################################################################
        async def websocket_listener(websocket):
            global user_message_queue
            while True:
                print("before websocket receive")                
                data = await websocket.receive_text()
                print("after websocket receive")  
                if data == "__ping__":
                    print("PLUTO2")
                    continue
                
                else:
                    print("👤 User responded:", data)
                    print("before message queue put")  
                    await user_message_queue.put(data)
                    print("after message queue put")
        
       
        async def wrapped_input_func(*args, **kwargs):
                global  user_message_queue
            
                print("⏳ Waiting for user input (moderator turn)...")
                while True:
                    print("before message queue get")
                    msg = await user_message_queue.get()
                    print("after message queue get")
                    if msg and msg.strip():
                        return msg         

##########################################################################################################
################################# user_proxy  #######################################################

        agents["user_proxy"] = UserProxyAgent(name="user_proxy", input_func=wrapped_input_func)
        
##########################################################################################################
################################# supervisor_agent  ####################################################### 
####### agents includes supervisor_agent
####### agent_list does NOT include supervisor_agent    
   
        agents["supervisor_agent"] = AssistantAgent(
            name="supervisor_agent",
            description="Monitors the debate and proposes agent behavior changes.",
            system_message=(
                "You are responsible for analyzing the ongoing debate and suggesting changes to any agent's system message "
                "or temperature. Always respond in JSON format like:\n"
                '{"agent_name": "expert_1_agent", "new_description": "Speak more clearly.", "new_temperature": 0.6}\n'
                "If no update is needed, respond with: null"
            ),
            model_client=model_client_openai
        )
     
        """
        agent_list = [
            agents["moderator_agent"],
            agents["expert_1_agent"],
            agents["expert_2_agent"],
            agents["hilarious_agent"],
            agents["creative_agent"],
            agents["user_proxy"],
        ]
        """
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        
        agent_list = []
        for json_key in config.keys():
            if json_key in agents:
                agent_list.append(agents[json_key])
                
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
        
        asyncio.create_task(websocket_listener(websocket))
        
        asyncio.create_task(speak_worker(websocket))
        
        ###############################################
        
        supervisor_task = asyncio.create_task(supervisor_agent_loop())
        
        ###############################################
        
        
        
        await run_chat(team, websocket=websocket)
        
        await speech_queue.join()

        stop_execution = True
        return "✅ Debate complete."

    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
        await speech_queue.put(("system", "TERMINATE"))
        stop_execution = True
        team = None

    except Exception as e:
        traceback.print_exc()
        await websocket.send_text("⚠️ Internal server error during debate.")

    finally:
        print("🧹 Cleaning up session state...")
        stop_execution = True
    
        try:
            # 🔕 Gracefully shut down speaker
            await speech_queue.put(("system", "TERMINATE"))
            print("🔚 'TERMINATE' sent to speak_worker. Waiting for queue to drain...")
            await speech_queue.join()
            print("✅ speak_worker queue fully drained.")
        except Exception as e:
            print("⚠️ Error while shutting down speak_worker:", e)
    
        # 🧼 Now safe to clear everything
        team = None
        task1 = None
        agent_list = []
        supervisor_task.cancel()
        user_message_queue = asyncio.Queue()
        speech_queue = asyncio.Queue()
    
        await websocket.close(code=1001)
