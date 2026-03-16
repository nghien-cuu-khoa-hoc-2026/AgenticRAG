from __future__ import annotations
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from backend.utils.config import Settings
load_dotenv()


GROQ_API_KEY = Settings.groq_api_key
GEMINI_API_KEY = Settings.gemini_api_key

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")


def get_groq_llm() -> ChatGroq:
    """
    Return the GROQ model.

    """
    model_name = "llama-3.3-70b-versatile"
    return ChatGroq(
        model=model_name,
        api_key=GROQ_API_KEY,
        temperature=0.1,
        streaming=True
    )


def get_gemini_llm() -> ChatGoogleGenerativeAI:
    """
    Return the GEMINI model.

    """
    model_name = "gemini-3-flash-preview"
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.1,
        max_tokens=8192,
        convert_system_message_to_human=True
    )
