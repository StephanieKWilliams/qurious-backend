"""
FastAPI Backend for Qurious Chat Application
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import os
from datetime import datetime
import httpx
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=" Chat API",
    description="Backend API for Qurious Chat Application with ultra-fast inference",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

ALLOWED_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_ENV.split(",") if origin.strip()]

if not ALLOWED_ORIGINS:
    print("⚠️ WARNING: No CORS origins defined! CORS requests may fail.")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Pydantic models
class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=4000)
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: Optional[List[Message]] = []
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()

class ChatResponse(BaseModel):
    response: str
    timestamp: datetime
    model: str = "llama-3.1-8b-instant"
    tokens_used: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    model: str = "llama-3.1-8b-instant"

# Dependency for API key validation
async def validate_api_key():
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Groq API key not configured"
        )
    return True

# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        model="llama-3.1-8b-instant"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        model="llama-3.1-8b-instant"
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    _: bool = Depends(validate_api_key)
):
    """
    Main chat endpoint that processes user messages and returns Qurious responses
    """
    try:
        logger.info(f"Processing chat request: {request.message[:50]}...")
        
        # Build conversation context
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant powered by Groq's ultra-fast inference. Provide clear, accurate, and helpful responses. Be conversational and friendly while maintaining professionalism. You excel at reasoning, coding, analysis, creative tasks, and general questions with lightning-fast response times."
            }
        ]
        
        # Add conversation history (limit to last 10 messages for context)
        if request.history:
            recent_history = request.history[-10:]
            for msg in recent_history:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": request.message
        })
        
        # Call Groq API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": messages,
                    "max_tokens": 1000,
                    "temperature": 0.7,
                    "top_p": 1,
                    "stream": False
                },
                timeout=30.0
            )
        
        if response.status_code == 401:
            logger.error("Groq API authentication failed")
            raise HTTPException(
                status_code=401,
                detail="Invalid Groq API key"
            )
        elif response.status_code == 429:
            logger.error("Groq API rate limit exceeded")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        elif response.status_code != 200:
            logger.error(f"Groq API error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Groq API error: {response.text}"
            )
        
        response_data = response.json()
        
        # Extract response
        ai_response = response_data["choices"][0]["message"]["content"]
        tokens_used = response_data.get("usage", {}).get("total_tokens", 0)
        
        logger.info(f"Generated response with {tokens_used} tokens")
        
        return ChatResponse(
            response=ai_response,
            timestamp=datetime.now(),
            model="llama-3.1-8b-instant",
            tokens_used=tokens_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@app.get("/api/models")
async def get_available_models(_: bool = Depends(validate_api_key)):
    """Get list of available Groq models"""
    return {
        "models": [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ],
        "default": "llama-3.1-8b-instant",
        "timestamp": datetime.now()
    }

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(
        status_code=400,
        detail=str(exc)
    )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return HTTPException(
        status_code=404,
        detail="Endpoint not found"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
