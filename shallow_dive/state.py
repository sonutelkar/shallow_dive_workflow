"""State definition for the Shallow Dive workflow."""

from typing import TypedDict, List, Dict, Any


class ShallowDiveState(TypedDict):
    """State object for the shallow dive analysis workflow."""

    # Input
    company_name: str
    ticker: str

    # Research Data
    company_overview: Dict[str, Any]
    financial_metrics: Dict[str, Any]
    ownership_data: Dict[str, Any]
    market_data: Dict[str, Any]
    web_research: List[Dict[str, Any]]

    # Source Tracking
    sources: List[Dict[str, Any]]
    source_map: Dict[str, int]

    # Analysis Sections - Profile
    section_1_1: str
    section_1_2: str
    section_1_3: str
    section_1_4: str
    section_1_5: str

    # Analysis Sections - Phase 1 Investment Analysis
    section_2_1: str  # Key value drivers & catalysts
    section_2_2: str  # Implied expectations
    section_2_3: str  # Key assumptions vs consensus

    # Analysis Sections - Industry
    section_3_1: str  # Profit pool & value chain
    section_3_2: str  # Five forces
    section_3_3: str  # Industry structure classification

    # Analysis Sections - Economic value-added generation
    section_4_1: str  # ROIC analysis
    section_4_2: str  # Competitive advantage (7 Powers)
    section_4_3: str  # Reinvestment opportunity
    section_4_4: str  # Sustainability of advantage

    # Risks
    section_5_1: str  # Risk identification & quantification
    section_5_2: str  # Pre-mortem scenarios

    # Valuation & positioning
    section_6_1: str  # Peer review
    section_6_2: str  # Relative valuation
    section_6_3: str  # 3-year price target

    # Identification (Stewardship vs Legacy)
    section_7_1: str

    # Engagement
    section_8_1: str  # Culture
    section_8_2: str  # Sustainability assessment
    section_8_3: str  # Risk mitigation vs peers
    section_8_4: str  # Engagement plan

    # Investment Recommendation
    investment_rating: str  # BUY / HOLD / SELL
    target_price: str
    upside_potential: str

    # Workflow Control
    current_section: str
    completed_sections: List[str]
    errors: List[str]

    # Final Output
    final_report: str
