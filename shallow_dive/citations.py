"""Citation helpers for tracking and rendering sources."""

from datetime import datetime
from typing import Dict, Any, List, Tuple

from .state import ShallowDiveState


def add_source(state: ShallowDiveState, url: str, title: str, content: str = "") -> Tuple[ShallowDiveState, int]:
    """Add a source and return its citation number."""
    if url in state.get("source_map", {}):
        return state, state["source_map"][url]

    citation_num = len(state.get("sources", [])) + 1
    source_entry: Dict[str, Any] = {
        "number": citation_num,
        "url": url,
        "title": title,
        "content_snippet": content[:200] if content else "",
        "accessed_date": datetime.now().strftime("%Y-%m-%d"),
    }

    state.setdefault("sources", []).append(source_entry)
    state.setdefault("source_map", {})[url] = citation_num
    return state, citation_num


def generate_references_section(state: ShallowDiveState) -> str:
    """Generate formatted references section."""
    if not state.get("sources"):
        return "No sources cited."

    references = ["## References\n"]
    for source in sorted(state["sources"], key=lambda x: x["number"]):
        ref = f"[{source['number']}] {source['title']}. "
        ref += f"Retrieved {source['accessed_date']}. "
        ref += f"{source['url']}\n"
        references.append(ref)

    return "\n".join(references)
