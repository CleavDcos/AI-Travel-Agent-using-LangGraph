"""
FastAPI Server for AI Travel Agent
Wraps existing LangGraph + Gradio setup with API endpoints
"""

import os
import uuid
import json
from typing import Optional, List, Dict, Any

# FastAPI & Uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Gradio
import gradio as gr

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# IMPORT EXISTING LANGGRAPH COMPONENTS
# ============================================================
# Note: In production, these would be imported from a separate module
# For this implementation, we replicate the essential setup

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
from typing import TypedDict, Annotated, Sequence
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import requests
from datetime import date

# ============================================================
# FASTAPI APP SETUP
# ============================================================

app = FastAPI(
    title="AI Travel Agent API",
    description="LangGraph-powered travel assistant with web search and hotel booking",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class ChatRequest(BaseModel):
    user_input: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    tools_used: List[str]
    thread_id: str

# ============================================================
# LANGGRAPH AGENT SETUP (from existing notebook)
# ============================================================

# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]

# Tool: Get Current Date
@tool
def get_current_date_tool():
    """Returns the current date in YYY-MM-DD format. Used to get dates for hotel or flight booking"""
    return date.today().isoformat()

# Tool: Hotel Search via RapidAPI
def search_hotels(city: str, checkin_date: str, checkout_date: str):
    """Search hotels using RapidAPI Booking.com"""
    RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
    
    url = "https://booking-com.p.rapidapi.com/v1/hotels/search"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
    }
    params = {
        "checkin_date": checkin_date,
        "checkout_date": checkout_date,
        "units": "metric",
        "adults_number": 2,
        "room_number": 1,
        "dest_type": "city",
        "dest_id": "-2092174",
        "order_by": "popularity",
        "locale": "en-gb"
    }

    if not RAPIDAPI_KEY:
        return [{"error": "RAPIDAPI_KEY is not set"}]

    response = requests.get(url, headers=headers, params=params, timeout=20)
    data = response.json()

    results = []
    if "result" in data:
        for hotel in data["result"][:5]:
            results.append({
                "name": hotel.get("hotel_name"),
                "price": hotel.get("min_total_price"),
                "rating": hotel.get("review_score"),
                "address": hotel.get("address")
            })

    return str(results)

@tool
def hotel_search_tool(city: str, checkin_date: str, checkout_date: str):
    """Search hotels using RapidAPI Booking.com"""
    return search_hotels(city, checkin_date, checkout_date)

# Tool: Tavily Web Search
tavily_search_tool = TavilySearchResults(max_results=3)

# ============================================================
# LANGGRAPH BUILD FUNCTIONS
# ============================================================

def make_call_model_with_tools(tools: list):
    """Create a node function that binds tools to LLM"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    def call_model_with_tools(state: AgentState):
        print("DEBUG: Entering call_model_with_tools node")
        messages = state["messages"]
        model_with_tools = llm.bind_tools(tools)
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    return call_model_with_tools

def should_continue(state: AgentState) -> str:
    """Conditional edge logic"""
    print("DEBUG: Entering should_continue node")
    last_message = state["messages"][-1]
    
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("DEBUG: Decision: continue (route to action)")
        return "action"
    else:
        print("DEBUG: Decision: end (route to END)")
        return END

def build_graph(tools: list):
    """Build the LangGraph workflow"""
    tool_node = ToolNode(tools)
    call_node_fn = make_call_model_with_tools(tools)
    
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_node_fn)
    graph.add_node("action", tool_node)
    graph.set_entry_point("agent")
    
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"action": "action", END: END}
    )
    graph.add_edge("action", "agent")
    
    return graph.compile()

# ============================================================
# BUILD THE TRAVEL AGENT
# ============================================================

tools = [hotel_search_tool, tavily_search_tool]
app_travel_agent = build_graph(tools)

# ============================================================
# CHAT FUNCTION (for both API and Gradio)
# ============================================================

def travel_agent_chat(user_input: str, history=None):
    """Streaming chat function for Gradio"""
    tools_used = []
    stream = app_travel_agent.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config={"recursion_limit": 15, "configurable": {"thread_id": str(uuid.uuid4())}},
    )

    for chunk in stream:
        _, node = next(iter(chunk.items()))
        if isinstance(node, dict) and "messages" in node:
            for msg in node["messages"]:
                if isinstance(msg, ToolMessage):
                    if msg.name not in tools_used:
                        tools_used.append(msg.name)
                    yield f"\n\n **Tool:** {msg.name}\n{msg.content}\n\n—\n\n"
                elif isinstance(msg, AIMessage) and msg.content:
                    yield msg.content

    if tools_used:
        yield f"\n\n**Tools used this session:** {', '.join(tools_used)} \n\n — \n\n {msg.content}"

# ============================================================
# NON-STREAMING API CHAT FUNCTION
# ============================================================

def chat_sync(user_input: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """Synchronous chat for API endpoint"""
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    tools_used = []
    final_response = ""
    
    stream = app_travel_agent.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config={"recursion_limit": 15, "configurable": {"thread_id": thread_id}},
    )

    for chunk in stream:
        _, node = next(iter(chunk.items()))
        if isinstance(node, dict) and "messages" in node:
            for msg in node["messages"]:
                if isinstance(msg, ToolMessage):
                    if msg.name not in tools_used:
                        tools_used.append(msg.name)
                elif isinstance(msg, AIMessage) and msg.content:
                    final_response += msg.content

    return {
        "response": final_response,
        "tools_used": tools_used,
        "thread_id": thread_id
    }

# ============================================================
# FASTAPI ENDPOINTS
# ============================================================

#@app.get("/")
#async def root():
 #   """Root endpoint - redirects to Gradio UI"""
  #  return {"message": "AI Travel Agent API", "ui": "Visit / to use Gradio interface"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Travel Agent API"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    API endpoint for chat interactions
    
    Args:
        user_input: The user's message
        thread_id: Optional thread ID for conversation continuity
    
    Returns:
        JSON with response, tools_used, and thread_id
    """
    try:
        result = chat_sync(request.user_input, request.thread_id)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming chat endpoint (SSE-like response)
    For true streaming, use Server-Sent Events
    """
    try:
        result = chat_sync(request.user_input, request.thread_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# GRADIO INTERFACE
# ============================================================

def gradio_chat(user_input, history=None):
    response = ""
    for chunk in travel_agent_chat(user_input, history):
        response += chunk
    return response

# Create Gradio Interface
gradio_interface = gr.ChatInterface(
    fn=gradio_chat,
    chatbot=gr.Chatbot(
        height=650,
        label="AI Travel Agent",
    ),
    textbox=gr.Textbox(
        placeholder="Plan your trip! Ask about attractions, travel advisories, and hotels...",
        container=False,
        scale=7
    ),
    title="✈️ LangGraph AI Travel Agent 🌍",
    description="Your stateful travel assistant with web search and hotel booking capabilities",
    examples=[
        ["What are the top 3 tourist attractions in Tokyo?"],
        ["Share the latest travel advisories for New York."],
        ["Find hotel options in Paris from 2026-06-01 to 2026-06-05."],
    ],
    cache_examples=False,
)

# ============================================================
# MOUNT GRADIO ON FASTAPI
# ============================================================

# Mount Gradio app at root path
gradio_app = gr.mount_gradio_app(app, gradio_interface, path="/")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )