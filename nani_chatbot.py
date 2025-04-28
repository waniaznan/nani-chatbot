import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from langdetect import detect
from fuzzywuzzy import fuzz

# Load environment variables from .env file
load_dotenv()

# Setup OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# CORS configuration for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# Smarter typo detection
def is_similar(input_text, keyword):
    return fuzz.token_sort_ratio(input_text.lower(), keyword.lower()) > 80

# Language detection
def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return "en"

# Summarize intent if message too long
async def summarize_intent(message: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Summarize customer's intent from this message in one short sentence."},
                {"role": "user", "content": message},
            ],
            temperature=0.5,
            max_tokens=50,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return message

# Construct OpenAI prompt
async def generate_response(message: str, language: str) -> str:
    prompt = f"You are Nani, a polite, helpful sales assistant for a frozen food business. Answer customer inquiries in {'Malay' if language == 'ms' else 'English'} clearly and assist in closing the sale. Be professional and friendly. Message: {message}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Make sure your account has access
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": message},
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Return more detailed error message
        return f"Sorry, an error occurred: {str(e)}"

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    user_message = chat_request.message
    language = detect_language(user_message)
    reply = await generate_response(user_message, language)
    return {"response": reply}

# Run this app locally with:
# uvicorn nani_chatbot:app --reload
