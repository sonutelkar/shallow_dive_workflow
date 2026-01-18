"""External data and search helpers."""

import json
from typing import Dict, Any, List

import requests

from . import config
from .citations import add_source
from .state import ShallowDiveState

logger = config.logger


def web_search(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Perform web search using Tavily."""
    try:
        results = config.search_tool.invoke({"query": query, "max_results": num_results})
        return results if isinstance(results, list) else []
    except Exception as exc:  # pragma: no cover - runtime logging
        logger.warning(f"Search error: {exc}")
        return []


def get_financial_data(ticker: str, api_key: str | None = None) -> Dict[str, Any]:
    """Fetch financial data from Financial Modeling Prep API (stable endpoints)."""
    if not api_key:
        return {}

    try:
        metrics_url = f"https://financialmodelingprep.com/stable/key-metrics?symbol={ticker}&apikey={api_key}"
        metrics = requests.get(metrics_url, timeout=20).json()

        ttm_url = f"https://financialmodelingprep.com/stable/key-metrics-ttm?symbol={ticker}&apikey={api_key}"
        metrics_ttm = requests.get(ttm_url, timeout=20).json()

        ratios_url = f"https://financialmodelingprep.com/stable/ratios?symbol={ticker}&apikey={api_key}"
        ratios = requests.get(ratios_url, timeout=20).json()

        income_url = f"https://financialmodelingprep.com/stable/income-statement?symbol={ticker}&limit=5&apikey={api_key}"
        income = requests.get(income_url, timeout=20).json()

        return {
            "metrics": metrics[:5] if isinstance(metrics, list) else [],
            "metrics_ttm": metrics_ttm if isinstance(metrics_ttm, dict) else metrics_ttm if isinstance(metrics_ttm, list) else [],
            "ratios": ratios[:5] if isinstance(ratios, list) else [],
            "income_statement": income[:5] if isinstance(income, list) else [],
        }
    except Exception as exc:  # pragma: no cover - runtime logging
        logger.warning(f"Financial data error: {exc}")
        return {}


def get_company_profile(ticker: str, api_key: str | None = None) -> Dict[str, Any]:
    """Get company profile data using stable endpoint."""
    if not api_key:
        return {}

    try:
        url = f"https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={api_key}"
        profile = requests.get(url, timeout=20).json()
        return profile[0] if isinstance(profile, list) and profile else {}
    except Exception as exc:  # pragma: no cover - runtime logging
        logger.warning(f"Profile error: {exc}")
        return {}
