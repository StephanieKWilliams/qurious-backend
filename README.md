
🧠 AI Chat Backend – FastAPI API for AI-Powered Conversations

A powerful and minimal FastAPI backend that powers a modern AI chat frontend. Designed for speed, clarity, and modular AI integration with Groq’s ultra-fast inference API.

    💡 The frontend (built with Next.js 15) is separately hosted and connects via API.

🚀 Features

    🧠 Groq AI Integration – Ultra-fast inference with LLaMA & Mixtral models

    ⚡ FastAPI – Blazing-fast Python backend with Pydantic validation

    🧪 API Testing – Built-in /health and / endpoints

    🧾 OpenAPI Docs – Auto-generated Swagger & ReDoc documentation

    🔐 Secure Key Handling – Groq key managed via .env

    🔄 CORS-Ready – For frontend-to-backend browser requests

🛠 Tech Stack

    FastAPI – Python web framework

    Groq API – LLM backend

    Uvicorn – ASGI server

    HTTPX – Async HTTP client

    Pydantic – Data validation and serialization

    python-dotenv – Environment configuration

📦 Installation
Prerequisites

    Python 3.8+

    A valid Groq API Key from https://console.groq.com

Clone & Setup

git clone <this-repo-url>
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

⚙️ Environment Variables

Create a .env file in the root of backend/ directory.

GROQ_API_KEY=gsk_your_groq_api_key_here
DEBUG=True
LOG_LEVEL=INFO
ALLOWED_ORIGINS=https://qurious-seven.vercel.app

    ALLOWED_ORIGINS must match your deployed frontend domain.

🚀 Running the Backend

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

or

python main.py

Your FastAPI server will be available at:

http://localhost:8000

🔧 API Endpoints
GET /

    ✅ Health check root

    ℹ️ Returns basic server info

GET /health

    ✅ Full backend health diagnostics

    Returns system status and model readiness

GET /api/models

    📜 Lists supported LLM models

POST /api/chat

    🤖 Sends user message and conversation history to LLM


📚 Supported Models

    llama-3.1-8b-instant (default)

    llama-3.1-70b-versatile

    mixtral-8x7b-32768

    gemma2-9b-it

🧪 Testing the API
Health Check

curl http://localhost:8000/health

Chat Endpoint

curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello","history":[]}'

