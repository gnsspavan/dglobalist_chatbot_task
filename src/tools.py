"""
Tools for the unicorn startups agent.
"""
from typing import Optional

from langchain_core.tools import tool

from .rag import search_companies, format_search_results


@tool
def search_companies_tool(
    query_text: str,
    primary_sector: Optional[str] = None,
    location: Optional[str] = None,
    country: Optional[str] = None,
    company_name: Optional[str] = None,
    stage: Optional[str] = None,
    n_results: int = 10,
) -> str:
    """
    Search the unicorn startups knowledge base by semantic query and optional filters.
    """
    n = max(1, min(20, int(n_results))) if n_results else 10
    query_text = (query_text or "").strip() or (company_name or "unicorn startup") or "unicorn startup"
    kwargs = {
        "query_text": query_text,
        "primary_sector": primary_sector,
        "location": location,
        "country": country,
        "company_name": company_name,
        "stage": stage,
        "n_results": n,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    results = search_companies(**kwargs)
    return format_search_results(results)
