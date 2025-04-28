import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import openai
from langdetect import detect
from fuzzywuzzy import fuzz

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS configuration for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# Helper function to correct typos (basic fuzzy matching)
def is_similar(input_text, keyword):
    return fuzz.partial_ratio(input_text.lower(), keyword.lower()) > 80

# Detect language for multilingual support
def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return "en"

# Construct OpenAI prompt
async def generate_response(message: str, language: str) -> str:
    prompt = f"You are Nani, a polite, helpful sales assistant for a frozen food business. Answer customer inquiries in {'Malay' if language == 'ms' else 'English'} clearly and assist in closing the sale. Be professional and friendly. Message: {message}"

    try:
        from openai import OpenAI

        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": message},
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    user_message = chat_request.message
    language = detect_language(user_message)
    reply = await generate_response(user_message, language)
    return {"response": reply}

# Run with: uvicorn nani_chatbot:app --reload
