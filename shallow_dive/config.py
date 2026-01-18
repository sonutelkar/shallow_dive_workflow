"""Configuration and shared objects for the Shallow Dive workflow."""

import logging
import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Fix Windows console encoding for Unicode support
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# Load environment variables once
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Logging setup
logger = logging.getLogger("shallow_dive")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)
logger.propagate = False


def validate_api_keys() -> list[str]:
    """Return a list of missing API keys required to run the workflow."""
    missing = []
    if not (OPENAI_API_KEY or OPENROUTER_API_KEY):
        missing.append("OPENAI_API_KEY or OPENROUTER_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    return missing


def create_llm():
    """Instantiate ChatOpenAI with OpenRouter when available, else OpenAI."""
    if OPENROUTER_API_KEY:
        return ChatOpenAI(
            model=OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3,
        )
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0.3, api_key=OPENAI_API_KEY)


# Instantiate shared tools
llm = create_llm()
search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)

# System prompt for the expert analyst
ANALYST_SYSTEM_PROMPT = """You are an Expert Equity Research Assistant writing for a demanding, long-term owner-investor. 
Your audience values mechanism over narrative, structural advantage over hype, and falsifiability over optimism.

CRITICAL OUTPUT STYLE:
1. Write in concise, eloquent paragraphs - NO BULLET POINTS
2. Every paragraph must follow: Claim + Evidence + Implication
3. Be dense and specific - avoid fluff
4. Use exact numbers, percentages, and timeframes
5. Focus on mechanisms and structural drivers

CITATION REQUIREMENTS:
1. When making factual claims, include citations in the format [X] where X is the source number
2. Place citations at the END of sentences or claims, before the period
3. Multiple citations for one claim: [1][2] or [1,2]
4. You will be provided with sources that already have citation numbers assigned
5. Only cite sources that are provided in the context - do not invent citation numbers
6. Integrate citations naturally into prose without disrupting readability

GOOD EXAMPLE:
"The company sustains gross margins of 85%, driven by its proprietary data lock-in which renders switching costs 
prohibitively high for enterprise clients [1], suggesting a durability of pricing power that the market currently underestimates. 
Management has consistently highlighted this competitive advantage in recent earnings calls [2][3]."

BAD EXAMPLES:
"The company has high margins." (No citation, no specificity)
"According to source [1], the margin is 85%." (Awkward citation placement)
"""
