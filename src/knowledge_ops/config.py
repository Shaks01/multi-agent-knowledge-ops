"""
Central place for LLM configuration.

Every agent in this project should get its LLM client from `get_llm()`
here rather than constructing one itself. That keeps provider/model choice
a one-file change as the project grows (see ROADMAP.md).
"""

import os
from typing import Optional

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if present.
load_dotenv()

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")


def get_llm(model: Optional[str] = None, temperature: float = 0.0):
    """Build and return a LangChain chat model client.

    Currently backed by Google Gemini (free tier, no credit card required).
    Swapping providers later (Anthropic, OpenAI, a local Ollama model)
    means changing this function, not every script that calls it.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key or api_key == "your-key-here":
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add "
            "your key (get a free one at https://aistudio.google.com/apikey)."
        )

    # Imported here so importing this module doesn't require the package
    # to be installed until an LLM is actually requested.
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        google_api_key=api_key,
    )
