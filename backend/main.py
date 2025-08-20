# minimal_main.py - Start with this basic version to test API key and basic functionality

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Setup FastAPI app
app = FastAPI(title="Interview Bot API - Minimal Test", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://interview-coach-chi.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic test endpoints
@app.get("/")
async def root():
    return {
        "message": "Interview Bot API is running", 
        "status": "healthy",
        "version": "0.1.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Interview Bot API is running"}

@app.get("/test-env")
async def test_env():
    """Test if environment variables are loaded"""
    return {
        "api_key_exists": bool(api_key),
        "api_key_length": len(api_key) if api_key else 0,
        "api_key_preview": api_key[:10] + "..." if api_key and len(api_key) > 10 else "Not found"
    }

@app.get("/test-openrouter")
async def test_openrouter():
    """Test OpenRouter API connection"""
    try:
        if not api_key:
            return {"success": False, "error": "API key not found"}
        
        # Import here to avoid issues if not installed
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        # Test with a simple, reliable free model
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model="meta-llama/llama-3.1-8b-instruct:free",
            openai_api_key=api_key
        )
        
        response = llm.invoke([HumanMessage(content="Hello, this is a test.")])
        
        return {
            "success": True,
            "model": "meta-llama/llama-3.1-8b-instruct:free",
            "response": response.content[:100] if response and response.content else "No response"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@app.get("/test-gpt-oss")
async def test_gpt_oss():
    """Test specifically GPT-OSS-20B model"""
    try:
        if not api_key:
            return {"success": False, "error": "API key not found"}
        
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        # Test your preferred model
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-oss-20b:free",
            openai_api_key=api_key
        )
        
        response = llm.invoke([HumanMessage(content="Hello, this is a test.")])
        
        return {
            "success": True,
            "model": "openai/gpt-oss-20b:free",
            "response": response.content[:100] if response and response.content else "No response"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

# Simple test models
class TestMessage(BaseModel):
    message: str

@app.post("/test-post")
async def test_post(data: TestMessage):
    """Test POST endpoint"""
    return {"received": data.message, "status": "success"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="localhost", port=port)