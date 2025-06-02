
import os
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from autogen import AssistantAgent

app = FastAPI()

# Enable CORS for frontend access (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agent configuration
assistant = AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]
    },
)

@app.get("/")
def root():
    return {"message": "Multi-agent API is live"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("user_message", "")
    result = await assistant.run(task=user_message)
    return {"reply": result.messages[-1].content if result.messages else "No reply"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            result = await assistant.run(task=data)
            reply_text = result.messages[-1].content if result.messages else "No reply"
            await websocket.send_text(reply_text)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
