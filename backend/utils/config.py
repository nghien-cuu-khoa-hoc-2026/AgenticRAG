from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


import os
from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class Settings:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RERANK_MODEL: Optional[str] = os.getenv("RERANK_MODEL")
    EMBEDDING_MODEL: Optional[str] = os.getenv("EMBEDDING_MODEL")
    chroma_host: Optional[str] = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: Optional[int] = int(os.getenv("CHROMA_PORT", 8080))
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    tavily_api_key: Optional[str] = os.getenv("TAVILY_API_KEY")
    firecrawl_api_key: Optional[str] = os.getenv("FIRECRAWL_API_KEY")
    serpapi_api_key: Optional[str] = os.getenv("SERPAPI_API_KEY")
    llama_index_api_key: Optional[str] = os.getenv("LLAMA_CLOUD_API_KEY")