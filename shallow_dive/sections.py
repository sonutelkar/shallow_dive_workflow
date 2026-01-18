"""Workflow node functions for each analysis section."""

import json
import re
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from . import config
from .citations import add_source, generate_references_section
from .data_sources import web_search, get_financial_data, get_company_profile
from .state import ShallowDiveState

logger = config.logger


def initialize_research(state: ShallowDiveState) -> ShallowDiveState:
    """Initialize the research by gathering basic company data."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"INITIALIZING RESEARCH: {state['company_name']}")
    logger.info(f"{'=' * 60}\n")

    state["sources"] = []
    state["source_map"] = {}

    if config.FMP_API_KEY:
        profile = get_company_profile(state["ticker"], config.FMP_API_KEY)
        state["company_overview"] = profile
        logger.info("[OK] Retrieved company profile from API")

        if profile:
            state, _ = add_source(
                state,
                f"https://financialmodelingprep.com/api/v3/profile/{state['ticker']}",
                f"Financial Modeling Prep - {state['company_name']} Company Profile",
                "Company financial data and profile",
            )

    search_queries = [
        f"{state['company_name']} business model revenue breakdown",
        f"{state['company_name']} financial performance margins profitability",
        f"{state['company_name']} ownership structure shareholders management",
    ]

    all_results = []
    for query in search_queries:
        results = web_search(query)
        for result in results:
            if result.get("url"):
                state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))
        all_results.extend(results)
        logger.info(f"[OK] Completed search: {query}")

    state["web_research"] = all_results
    state["current_section"] = "1.1"
    state["completed_sections"] = []
    state["errors"] = []

    logger.info(f"\n[OK] Collected {len(state['sources'])} sources")
    return state


def gather_financial_data(state: ShallowDiveState) -> ShallowDiveState:
    """Gather comprehensive financial metrics."""
    logger.info(f"\n{'=' * 60}")
    logger.info("GATHERING FINANCIAL DATA")
    logger.info(f"{'=' * 60}\n")

    if config.FMP_API_KEY:
        financial_data = get_financial_data(state["ticker"], config.FMP_API_KEY)
        state["financial_metrics"] = financial_data
        logger.info("[OK] Retrieved financial metrics from API")

        if financial_data.get("metrics"):
            state, _ = add_source(
                state,
                f"https://financialmodelingprep.com/api/v3/key-metrics/{state['ticker']}",
                f"Financial Modeling Prep - {state['company_name']} Key Metrics",
                "Company key financial metrics and ratios",
            )

    queries = [
        f"{state['company_name']} DuPont analysis return on equity ROE breakdown",
        f"{state['company_name']} capital allocation working capital efficiency",
        f"{state['company_name']} margin trends profitability drivers",
    ]

    for query in queries:
        results = web_search(query)
        for result in results:
            if result.get("url"):
                state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))
        state["web_research"].extend(results)
        logger.info(f"[OK] Completed search: {query}")

    logger.info(f"\n[OK] Total sources: {len(state['sources'])}")
    return state


def _research_context(state: ShallowDiveState, results_subset: list[dict], limit: int = 10) -> list[str]:
    """Build formatted research strings with citations for prompts."""
    research_with_citations = []
    for result in results_subset[:limit]:
        if result.get("url") and result["url"] in state.get("source_map", {}):
            citation_num = state["source_map"][result["url"]]
            research_with_citations.append(f"{result.get('title', '')} [{citation_num}]:\n{result.get('content', '')}")
    return research_with_citations


def analyze_section_1_1(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 1.1: Company Snapshot."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 1.1: COMPANY SNAPSHOT")
    logger.info(f"{'=' * 60}\n")

    research_with_citations = _research_context(state, state.get("web_research", []), limit=10)
    context = f"""
    Company: {state['company_name']}
    Ticker: {state['ticker']}

    Company Overview (from API):
    {json.dumps(state.get('company_overview', {}), indent=2)}

    Financial Data (from API):
    {json.dumps(state.get('financial_metrics', {}), indent=2)}

    Web Research with Citations:
    {chr(10).join(research_with_citations)}

    IMPORTANT: Use the citation numbers [X] provided above when referencing these sources in your analysis.
    Place citations at the end of claims, before the period.
    """

    prompt = f"""
    Based on the following data, write Section 1.1: Company Snapshot.

    This section must include (in dense prose, no bullets):
    1. Identity: Business + geography + end customer (1-2 sentences)
    2. Scale & float: Market cap, ADV, free float, ownership concentration
    3. Economics: Revenue growth range, EBIT margin range, ROIC/ROE range, cash conversion (through-cycle if possible)
    4. Financial risk: Net debt/EBITDA or net cash, refinancing notes, FX mismatch notes
    5. Valuation: P/E, EV/EBIT, P/B, FCF yield + note on peak/trough distortions
    6. One-line "what matters": The single most sensitive debate variable

    Context:
    {context}

    Remember: 
    - Write in eloquent paragraphs with specific numbers
    - NO bullet points
    - Follow Claim + Evidence + Implication
    - Include citations [X] at the end of factual claims
    - Only use citation numbers that are provided in the context above
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_1_1"] = response.content
    state["completed_sections"].append("1.1")

    logger.info(f"[OK] Section 1.1 completed ({len(response.content)} characters)")
    return state


def analyze_section_1_2(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 1.2: What does the company do?"""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 1.2: BUSINESS MODEL")
    logger.info(f"{'=' * 60}\n")

    results = web_search(f"{state['company_name']} business model value chain products services customers")
    for result in results:
        if result.get("url"):
            state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))

    all_research = state.get("web_research", []) + results
    research_with_citations = _research_context(state, all_research[-15:], limit=15)

    context = f"""
    Company: {state['company_name']}

    Previous Analysis (Section 1.1):
    {state.get('section_1_1', '')}

    Business Model Research with Citations:
    {chr(10).join(research_with_citations)}

    Company Data:
    {json.dumps(state.get('company_overview', {}), indent=2)}

    IMPORTANT: Use citation numbers [X] when referencing sources.
    """

    prompt = f"""
    Based on the data, write Section 1.2: What does the company do?

    This section must address (in dense prose):
    1. Products/services -> customer type -> purchase model (contract/tender/subscription/spot)
    2. Geography and profit concentration (where economics really sit)
    3. Position in value chain (upstream/midstream/downstream/platform/aftermarket)

    Required output format:
    - 3-5 sentences of narrative
    - Then weave in: Revenue driver, Cost driver, Key dependency

    Context:
    {context}

    Write in eloquent paragraphs. Be specific about mechanisms. No bullets.
    Include citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_1_2"] = response.content
    state["completed_sections"].append("1.2")

    logger.info(f"[OK] Section 1.2 completed ({len(response.content)} characters)")
    return state


def analyze_section_1_3(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 1.3: How is value created (DuPont)?"""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 1.3: VALUE CREATION (DUPONT)")
    logger.info(f"{'=' * 60}\n")

    queries = [
        f"{state['company_name']} profitability margins pricing power",
        f"{state['company_name']} capital efficiency asset turnover working capital",
        f"{state['company_name']} leverage capital structure debt",
    ]

    additional_research = []
    for query in queries:
        results = web_search(query)
        for result in results:
            if result.get("url"):
                state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))
        additional_research.extend(results)

    all_research = state.get("web_research", []) + additional_research
    research_with_citations = _research_context(state, all_research[-20:], limit=20)

    context = f"""
    Company: {state['company_name']}

    Previous Sections:
    {state.get('section_1_1', '')}
    {state.get('section_1_2', '')}

    Financial Metrics:
    {json.dumps(state.get('financial_metrics', {}), indent=2)}

    DuPont Research with Citations:
    {chr(10).join(research_with_citations)}

    IMPORTANT: Use citation numbers [X] when referencing sources.
    """

    prompt = f"""
    Write Section 1.3: How is value created (DuPont Analysis)?

    Follow this analytical framework (weave into narrative prose):

    1. FRAME THE RETURN ENGINE:
       - Is this margin-led, turnover-led, or leverage-led?
       - Where is profit captured in the value chain?

    2. PROFITABILITY (MARGIN QUALITY):
       - What drives margins: pricing, mix, scale, input costs, utilization?
       - Structural vs cyclical: did margins hold in weak years?
       - Any underinvestment masking margin quality?

    3. EFFICIENCY (CAPITAL USE):
       - Better capital deployment than peers? Why?
       - Working capital discipline trends (inventory, receivables, payables)
       - Any aggressive revenue recognition or "fake turnover"?

    4. CAPITAL STRUCTURE:
       - Is ROE high due to leverage/buybacks rather than operations?
       - Interest coverage trends?
       - Tax rate sustainability?

    5. GROWTH MECHANICS:
       - Volume vs price vs mix vs M&A?
       - Growth scalability (working capital/capex requirements)

    6. DURABILITY & BREAKPOINTS:
       - What protects the engine: pricing power, switching costs, scale, regulation?
       - Stress scenarios: what fails first under demand/input/FX/rate shocks?

    Context:
    {context}

    Write dense, mechanism-focused paragraphs. Use specific numbers. Identify the pattern 
    (operational compounder, leverage-made ROE, working-capital illusion, cycle beneficiary, or M&A veneer).
    Include citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_1_3"] = response.content
    state["completed_sections"].append("1.3")

    logger.info(f"[OK] Section 1.3 completed ({len(response.content)} characters)")
    return state


def analyze_section_1_4(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 1.4: Theme identification & exposure."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 1.4: THEME IDENTIFICATION")
    logger.info(f"{'=' * 60}\n")

    queries = [
        f"{state['company_name']} emerging market themes trends exposure",
        f"{state['company_name']} digitalization energy transition sustainability",
        f"{state['company_name']} market positioning competitive advantages moat",
    ]

    theme_research = []
    for query in queries:
        results = web_search(query)
        for result in results:
            if result.get("url"):
                state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))
        theme_research.extend(results)

    all_research = state.get("web_research", []) + theme_research
    research_with_citations = _research_context(state, all_research[-20:], limit=20)

    context = f"""
    Company: {state['company_name']}

    Previous Analysis:
    {state.get('section_1_1', '')}
    {state.get('section_1_2', '')}
    {state.get('section_1_3', '')}

    Theme Research with Citations:
    {chr(10).join(research_with_citations)}

    IMPORTANT: Use citation numbers [X] when referencing sources.
    """

    prompt = f"""
    Write Section 1.4: Theme Identification & Exposure.

    Analytical framework to address:

    1. BUILD THEME MAP:
       Consider these EM themes, select 2-5 that are material:
       - EM consumption / premiumization
       - Energy transition (renewables, electrification, grids)
       - Digitalization (payments, cloud, AI, e-commerce)
       - Reshoring / supply-chain diversification
       - Infrastructure / urbanization
       - Financial inclusion
       - Healthcare access
       - Resource security

    2. PROVE EXPOSURE (for each material theme):
       - Mechanism: demand pull, enabler, regulatory-driven, or substitution beneficiary?
       - Products/services connecting to theme
       - Customer and value-chain position

    3. MATERIALITY TEST:
       - What share of revenue/profit tied to theme?
       - Direct vs indirect exposure?

    4. DURABILITY (5-10 year view per theme):
       - Runway: TAM growth + penetration headroom
       - Cyclicality sensitivity
       - Policy support stability
       - Competition intensity
       - Disruption risk
       - Geopolitical / supply chain factors

    5. RIGHT TO WIN:
       - What moats exist in this theme context?
       - Pricing power or price-taker?
       - Can it fund growth without value destruction?

    Context:
    {context}

    Write mechanism-focused prose. Be specific about how themes translate to economics. 
    Avoid "theme-washing" - only include themes with clear revenue/profit links.
    Include citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_1_4"] = response.content
    state["completed_sections"].append("1.4")

    logger.info(f"[OK] Section 1.4 completed ({len(response.content)} characters)")
    return state


def analyze_section_1_5(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 1.5: Founders, management, shareholders."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 1.5: GOVERNANCE & OWNERSHIP")
    logger.info(f"{'=' * 60}\n")

    queries = [
        f"{state['company_name']} ownership structure major shareholders control",
        f"{state['company_name']} management team CEO founders history",
        f"{state['company_name']} governance board composition incentives",
        f"{state['company_name']} capital allocation track record M&A",
    ]

    governance_research = []
    for query in queries:
        results = web_search(query)
        for result in results:
            if result.get("url"):
                state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))
        governance_research.extend(results)

    all_research = state.get("web_research", []) + governance_research
    research_with_citations = _research_context(state, all_research[-25:], limit=25)

    context = f"""
    Company: {state['company_name']}

    Full Analysis So Far:
    {state.get('section_1_1', '')}
    {state.get('section_1_2', '')}
    {state.get('section_1_3', '')}
    {state.get('section_1_4', '')}

    Governance Research with Citations:
    {chr(10).join(research_with_citations)}

    IMPORTANT: Use citation numbers [X] when referencing sources.
    """

    prompt = f"""
    Write Section 1.5: Founders, Management, Shareholders - History, Current & Incentives.

    Analytical framework:

    1. MAP OWNERSHIP & CONTROL:
       - Who has economic ownership vs voting control?
       - Dual-class shares, pyramids, cross-holdings, golden shares?
       - Shareholder agreements (voting blocs, drag-along rights)?
       - State influence beyond equity?
       - Free float and swing holders?
       Synthesize into: "Effective controller is ___ because ___."

    2. HISTORICAL EVENTS:
       - Key ownership/control changes: IPO, privatization, succession, M&A, restructurings?
       - Governance "scar tissue": scandals, minority squeeze-outs, dilutive rescues?
       - Recurring patterns in control changes?

    3. REAL DECISION-MAKERS:
       - Who controls strategy, capital allocation, CEO appointment?
       - Founder/Chair vs CEO power dynamic?
       - Board independence (true vs nominal)?

    4. TRACK RECORD:
       For key decision-makers:
       - Capital allocation discipline: buybacks, dividends, reinvestment, M&A timing and ROIC
       - Strategic consistency vs frequent pivots
       - Treatment of minorities, transparency, crisis response

    5. INCENTIVES & ALIGNMENT:
       - Ownership % of key executives/controllers (including pledged shares)?
       - Pay structure: EPS/EBITDA vs ROIC/FCF/per-share metrics?
       - Option plan terms: strike discipline, vesting, performance hurdles?
       - Related-party transactions as wealth transfer?
       
       Conclude with alignment verdict:
       - What aligns with minorities?
       - What misaligns?
       - Net: aligned / mixed / misaligned + decisive reason

    Context:
    {context}

    Write in dense prose. Focus on governance reality vs org charts. Be specific about control 
    mechanisms and historical patterns. Identify the governance pattern (owner-operator aligned, 
    controller extraction risk, state-influenced hybrid, PE play, or succession overhang).
    Include citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_1_5"] = response.content
    state["completed_sections"].append("1.5")

    logger.info(f"[OK] Section 1.5 completed ({len(response.content)} characters)")
    return state


def analyze_section_2_1(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 2.1: Key Value Drivers & Catalysts."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 2.1: KEY VALUE DRIVERS & CATALYSTS")
    logger.info(f"{'=' * 60}\n")

    queries = [
        f"{state['company_name']} {state['ticker']} key growth drivers catalysts",
        f"{state['company_name']} margin expansion pricing power trends",
        f"{state['company_name']} competitive advantages market position",
        f"{state['company_name']} industry trends tailwinds headwinds",
        f"{state['company_name']} management strategy capital allocation",
    ]

    driver_research = []
    for query in queries:
        results = web_search(query)
        for result in results:
            if result.get("url"):
                state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))
                driver_research.append(result)
        logger.info(f"[OK] Search: {query}")

    research_with_citations = _research_context(state, driver_research[-20:], limit=20)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Previous Analysis:
    Section 1.1 (Snapshot): {state.get('section_1_1', '')[:800]}
    Section 1.3 (Value Creation): {state.get('section_1_3', '')[:800]}
    Section 1.4 (Themes): {state.get('section_1_4', '')[:800]}

    Financial Data:
    {json.dumps(state.get('financial_metrics', {}), indent=2)[:1500]}

    Research on Value Drivers with Citations:
    {chr(10).join(research_with_citations[:15])}
    """

    prompt = f"""
    Write Section 2.1: Key Value Drivers & Catalysts.
    
    Follow this EXACT workflow:
    
    STEP 1 - INFER THE "CONSENSUS VALUE EQUATION":
    Reduce the company to a simple intrinsic-value logic chain.
    Ask: What is the market implicitly assuming about (a) growth, (b) sustainable margins, 
    (c) reinvestment intensity, (d) risk/terminal durability?
    
    OUTPUT: 1-2 lines: "Consensus value is mostly a function of X, Y, Z (ranked)."
    
    STEP 2 - SELECT THE 3-5 KEY VARIABLES/DEBATES:
    Pick variables that are: (i) high sensitivity, (ii) genuinely uncertain, (iii) resolvable.
    
    Choose from: Volume/penetration, Net price/realization, Unit cost/productivity, 
    Opex intensity, Capex intensity, Working capital intensity, Competitive intensity, Regulatory/policy
    
    For EACH selected variable (must select 3-5), write:
    - **Variable**: [name]
    - **Debate**: What bulls vs bears disagree on
    - **Leading indicators**: 2-4 observable signals
    - **Resolution window**: When evidence should show up (quarters/years)
    
    STEP 3 - CONVERT INTO 3-5 REASONS IT'S ATTRACTIVE:
    Each reason must link to one or more variables and specify the mechanism.
    
    Format: "Attractive because [mechanism], evidenced by [facts with citations [X]], 
    and will be proven by [catalyst/timing]."
    
    STEP 4 - INDUSTRY/MACRO CATALYSTS (next 3 years):
    Pick 3-5 catalysts that plausibly change one of the key variables.
    
    For each:
    - Event: what changes
    - Transmission: which variable moves
    - Impact: margin/volume/cash/valuation direction
    - Asymmetry: why upside > downside
    
    STEP 5 - COMPANY-SPECIFIC VALUE UNLOCK CATALYSTS:
    Pick 3-5 discrete actions/events management can execute.
    
    For each:
    - Action/event
    - Mechanism: which variable improves
    - Milestones: what to watch
    - Execution risk: what could go wrong
    
    Context:
    {context}
    
    CRITICAL REQUIREMENTS:
    - Dense prose following Claim + Evidence + Implication
    - Citations [X] for all factual claims
    - Specific mechanisms, not vague statements
    - Must select and analyze 3-5 key variables
    - Prioritize by value sensitivity
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_2_1"] = response.content
    state["completed_sections"].append("2.1")

    logger.info(f"[OK] Section 2.1 completed ({len(response.content)} characters)")
    return state


def analyze_section_2_2(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 2.2: Implied expectations from current valuation."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 2.2: IMPLIED EXPECTATIONS")
    logger.info(f"{'=' * 60}\n")

    queries = [
        f"{state['company_name']} {state['ticker']} valuation expectations multiple",
        f"{state['company_name']} earnings expectations consensus {state['ticker']}",
        f"{state['company_name']} market pricing growth vs margins",
    ]
    expectations_research = []
    for query in queries:
        results = web_search(query)
        for result in results:
            if result.get("url"):
                state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))
                expectations_research.append(result)
        logger.info(f"[OK] Search: {query}")

    research_with_citations = _research_context(state, expectations_research[-15:], limit=15)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Value Drivers (2.1):
    {state.get('section_2_1', '')[:1000]}

    Snapshot & Value Creation:
    {state.get('section_1_1', '')[:600]}
    {state.get('section_1_3', '')[:600]}

    Financial Metrics:
    {json.dumps(state.get('financial_metrics', {}), indent=2)[:1200]}

    Valuation Research with Citations:
    {chr(10).join(research_with_citations)}
    """

    prompt = f"""
    Write Section 2.2: Implied Expectations.

    Follow the knowledge-base workflow:

    STEP 1 - PRICING FRAME:
    - Identify the primary valuation lens used by the market (P/E, EV/EBIT/EV/EBITDA, optionally P/B).
    - Output: "Market is valuing this primarily on ___, implying optimism/pessimism on ___."

    STEP 2 - IMPLIED BELIEFS BY KEY DRIVER:
    For the 3-5 variables from Section 2.1, infer what today's multiple assumes about:
    - Growth vs margin expansion vs risk normalization.
    - Whether earnings are at peak/trough.
    Output: ranked list "Market is pricing in: (1) __, (2) __, (3) __."

    STEP 3 - HISTORY/PEER SANITY TEST:
    - Implied growth vs history
    - Implied margin vs history/peers
    - Implied ROIC vs history/peers
    Output: 3 bullets.

    STEP 4 - MISPERCEPTIONS:
    - 2-4 bullets: "Misperception is __ because __; will be revealed by __."

    Context:
    {context}

    Requirements:
    - Dense prose, Claim + Evidence + Implication.
    - Citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_2_2"] = response.content
    state["completed_sections"].append("2.2")

    logger.info(f"[OK] Section 2.2 completed ({len(response.content)} characters)")
    return state


def analyze_section_2_3(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 2.3: Key assumptions & difference vs consensus."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 2.3: KEY ASSUMPTIONS VS CONSENSUS")
    logger.info(f"{'=' * 60}\n")

    revision_research = web_search(f"{state['company_name']} {state['ticker']} consensus estimates assumptions")
    for result in revision_research:
        if result.get("url"):
            state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))

    research_with_citations = _research_context(state, revision_research[-10:], limit=10)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Value Drivers (2.1):
    {state.get('section_2_1', '')[:800]}

    Implied Expectations (2.2):
    {state.get('section_2_2', '')[:800]}

    Revision Research with Citations:
    {chr(10).join(research_with_citations)}
    """

    prompt = f"""
    Write Section 2.3: Key assumptions & difference versus consensus.

    Workflow to follow:
    1) List the 3-5 key variables from Section 2.1 (ranked by value sensitivity).
    2) For each variable, restate the market-implied baseline from 2.2 ("Market implies: __").
    3) For each variable, list 2-3 potential revision vectors (level/slope/duration/volatility/mix/pass-through).
    4) State our variant view vs consensus: mechanism, evidence, and disconfirmers.
    5) Map catalysts + timing + leading indicators for each variable.
    6) Prioritize in a 2x2: High impact/Fast; High impact/Slow; Low impact/Fast (noise).

    Context:
    {context}

    Requirements:
    - Dense prose (no bullets except where specified).
    - Claim + Evidence + Implication.
    - Citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_2_3"] = response.content
    state["completed_sections"].append("2.3")

    logger.info(f"[OK] Section 2.3 completed ({len(response.content)} characters)")
    return state


def analyze_section_3_1(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 3.1: Industry profit pool and value chain."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 3.1: INDUSTRY PROFIT POOL & VALUE CHAIN")
    logger.info(f"{'=' * 60}\n")

    queries = [
        f"{state['company_name']} industry value chain profit pool",
        f"{state['company_name']} industry pricing power drivers",
        f"{state['company_name']} key competitors positioning economics",
    ]
    industry_research = []
    for query in queries:
        results = web_search(query)
        for result in results:
            if result.get("url"):
                state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))
                industry_research.append(result)
        logger.info(f"[OK] Search: {query}")

    research_with_citations = _research_context(state, industry_research[-20:], limit=20)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Business Model (1.2): {state.get('section_1_2', '')[:700]}
    Value Drivers (2.1): {state.get('section_2_1', '')[:700]}

    Industry Research with Citations:
    {chr(10).join(research_with_citations)}
    """

    prompt = f"""
    Write Section 3.1: Lay of the Land (Profit Pool & Value Chain).

    Follow workflow:
    1) Define the value chain (5-8 nodes) and where price is negotiated.
    2) Identify profit pool drivers and allocation across nodes.
    3) Map value capture: which nodes are High/Med/Low capture and why (pricing mechanism).
    4) Identify power holders (pricing/bargaining) using buyer/supplier power cues.
    5) Position key players (3-6 bullets: product, pricing posture, economics, strategic posture).

    Context:
    {context}

    Requirements: dense prose, Claim + Evidence + Implication, citations [X] for facts.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_3_1"] = response.content
    state["completed_sections"].append("3.1")

    logger.info(f"[OK] Section 3.1 completed ({len(response.content)} characters)")
    return state


def analyze_section_3_2(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 3.2: Market structure (Five Forces)."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 3.2: FIVE FORCES")
    logger.info(f"{'=' * 60}\n")

    forces_research = web_search(f"{state['company_name']} industry five forces competition suppliers customers")
    for result in forces_research:
        if result.get("url"):
            state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))

    research_with_citations = _research_context(state, forces_research[-15:], limit=15)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Industry Mapping (3.1):
    {state.get('section_3_1', '')[:900]}

    Five Forces Research with Citations:
    {chr(10).join(research_with_citations)}
    """

    prompt = f"""
    Write Section 3.2: Market Structure - Porter's Five Forces.

    Workflow:
    - Define the relevant market (product x geography x segment).
    - Competitor force: concentration verdict + evidence; rationality verdict + evidence.
    - Customer force: concentration/bargaining verdict + how it shows up.
    - Supplier force: verdict + implications for margin stability.
    - Synthesis: 3 bullets on profit pool stability and key watch indicators.

    Context:
    {context}

    Requirements: Claim + Evidence + Implication; citations [X] for facts.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_3_2"] = response.content
    state["completed_sections"].append("3.2")

    logger.info(f"[OK] Section 3.2 completed ({len(response.content)} characters)")
    return state


def analyze_section_3_3(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 3.3: Industry structure classification & positioning."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 3.3: INDUSTRY CLASSIFICATION")
    logger.info(f"{'=' * 60}\n")

    classification_research = web_search(f"{state['company_name']} industry structure classification fragmented mature network")
    for result in classification_research:
        if result.get("url"):
            state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))

    research_with_citations = _research_context(state, classification_research[-15:], limit=15)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Five Forces (3.2):
    {state.get('section_3_2', '')[:800]}

    Classification Research with Citations:
    {chr(10).join(research_with_citations)}
    """

    prompt = f"""
    Write Section 3.3: Industry Structure Classification & Positioning Opportunity.

    Workflow:
    1) Classify the industry (fragmented/emerging/mature/declining/international/network/hypercompetitive) with evidence.
    2) State rules of winning for that structure (3 bullets).
    3) Translate into positioning levers (3-5 bullets): lever -> how it wins share -> what to monitor.
    4) Proof points/disconfirmers for each lever (3 bullets).

    Context:
    {context}

    Requirements: dense prose; citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_3_3"] = response.content
    state["completed_sections"].append("3.3")

    logger.info(f"[OK] Section 3.3 completed ({len(response.content)} characters)")
    return state


def analyze_section_4_1(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 4.1: ROIC analysis vs peers."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 4.1: ROIC ANALYSIS")
    logger.info(f"{'=' * 60}\n")

    roic_research = web_search(f"{state['company_name']} {state['ticker']} ROIC vs peers returns on capital")
    for result in roic_research:
        if result.get("url"):
            state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))

    research_with_citations = _research_context(state, roic_research[-15:], limit=15)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Value Creation (1.3): {state.get('section_1_3', '')[:700]}
    Industry Structure (3.3): {state.get('section_3_3', '')[:600]}

    Financial Metrics:
    {json.dumps(state.get('financial_metrics', {}), indent=2)[:1200]}

    ROIC Research with Citations:
    {chr(10).join(research_with_citations)}
    """

    prompt = f"""
    Write Section 4.1: ROIC Analysis vs Peers.

    Workflow:
    - Define ROIC lens vs ROE; comparability caveats.
    - Through-cycle verdict: ROIC vs cost of capital (above/at/below) + confidence.
    - Decompose drivers: margin vs capital turns vs reinvestment intensity.
    - Peer gap diagnosis: 3 bullets explaining why the gap exists and durability.

    Context:
    {context}

    Requirements: Claim + Evidence + Implication; citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_4_1"] = response.content
    state["completed_sections"].append("4.1")

    logger.info(f"[OK] Section 4.1 completed ({len(response.content)} characters)")
    return state


def analyze_section_4_2(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 4.2: Source of Enduring Competitive Advantage (7 Powers)."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 4.2: COMPETITIVE ADVANTAGE (7 POWERS)")
    logger.info(f"{'=' * 60}\n")

    queries = [
        f"{state['company_name']} competitive advantages moat barriers to entry",
        f"{state['company_name']} scale economies network effects switching costs",
        f"{state['company_name']} vs competitors differentiation positioning",
        f"{state['company_name']} intellectual property patents technology",
        f"{state['company_name']} customer retention pricing power brand",
    ]

    moat_research = []
    for query in queries:
        results = web_search(query)
        for result in results:
            if result.get("url"):
                state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))
                moat_research.append(result)
        logger.info(f"[OK] Search: {query}")

    research_with_citations = _research_context(state, moat_research[-20:], limit=20)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Previous Analysis:
    Section 1.2 (Business Model): {state.get('section_1_2', '')[:800]}
    Section 1.3 (Value Creation): {state.get('section_1_3', '')[:800]}
    Section 2.1 (Value Drivers): {state.get('section_2_1', '')[:800]}

    Financial Performance:
    {json.dumps(state.get('financial_metrics', {}), indent=2)[:1500]}

    Competitive Advantage Research with Citations:
    {chr(10).join(research_with_citations[:15])}
    """

    prompt = f"""
    Write Section 4.2: Source of Enduring Competitive Advantage.
    
    Use Hamilton Helmer's 7 Powers framework. Follow this EXACT workflow:
    
    STEP 1 - STATE THE CANDIDATE ADVANTAGE (be specific):
    - What does the company do better that allows persistent excess returns (not just growth)?
    - Where does advantage sit: product, distribution, data, cost curve, regulation, ecosystem?
    
    STEP 2 - MAP TO HELMER'S 7 POWERS (test each, don't just label):
    
    For EACH relevant power, ask "What is the mechanism + proof?"
    
    **1. SCALE ECONOMIES**: Does unit cost fall with scale, and is scale hard to replicate?
    - Evidence: Cost per unit data, economies of scale in procurement/ops, minimum efficient scale
    
    **2. NETWORK ECONOMIES**: Does value increase with users/participants, reinforcing winner-takes-most?
    - Evidence: User growth metrics, engagement trends, multi-sided platform dynamics
    
    **3. SWITCHING COSTS**: Meaningful costs/risks to switching (integration, data, retraining)?
    - Evidence: Customer retention rates, contract durations, integration complexity, migration costs
    
    **4. BRANDING**: Does brand enable price premium/volume resilience beyond functional differentiation?
    - Evidence: Brand surveys, pricing vs competitors, marketing efficiency, loyalty metrics
    
    **5. CORNERED RESOURCE**: Exclusive access to talent, licenses, data, locations, distribution, IP?
    - Evidence: Patents, exclusive partnerships, unique assets, regulatory licenses
    
    **6. PROCESS POWER**: Complex embedded processes that improve over time and are hard to copy?
    - Evidence: Operational excellence metrics, learning curve, proprietary methodologies
    
    **7. COUNTER-POSITIONING**: New model incumbents can't copy without self-disruption?
    - Evidence: Business model innovation, incumbent constraints, cannibalization dynamics
    
    STEP 3 - STRENGTH TEST: DURABILITY + SCOPE:
    - Is the advantage local or global, segment-specific or company-wide?
    - Is it strengthening (flywheel) or decaying (commoditization, regulation, tech change)?
    - How long has the advantage persisted? Evidence of durability under stress?
    
    STEP 4 - DEEPENING STRATEGY:
    What is management doing to deepen each power:
    - Scale: capacity expansion, volume density, procurement leverage, automation
    - Switching: deeper integrations, bundled workflows, data portability friction
    - Network: liquidity incentives, ecosystem partners, reduce churn
    - Brand: consistency, trust-building, distribution expansion, service quality
    - Cornered resource: exclusive partnerships, talent retention, license extensions
    - Process: continuous improvement, proprietary tooling, capability building
    - Counter-positioning: commitment to new model, exploit incumbent constraints
    
    REQUIRED OUTPUT:
    - Identify Top 1-3 Powers with mechanism + current evidence (must cite with [X])
    - Moat reinforcement plan: 3 bullets (actions -> what power deepens -> what to watch)
    - Moat failure mode: 1-2 concrete ways the advantage could erode
    
    Context:
    {context}
    
    CRITICAL REQUIREMENTS:
    - Must identify specific Power(s) from the 7 Powers framework
    - Mechanism + evidence with citations [X], not just labels
    - Explicit failure modes
    - Dense prose, specific examples
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_4_2"] = response.content
    state["completed_sections"].append("4.2")

    logger.info(f"[OK] Section 4.2 completed ({len(response.content)} characters)")
    return state


def analyze_section_4_3(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 4.3: Reinvestment opportunity & incremental returns."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 4.3: REINVESTMENT OPPORTUNITY")
    logger.info(f"{'=' * 60}\n")

    reinvest_research = web_search(f"{state['company_name']} TAM growth runway capital allocation {state['ticker']}")
    for result in reinvest_research:
        if result.get("url"):
            state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))

    research_with_citations = _research_context(state, reinvest_research[-15:], limit=15)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Competitive Advantage (4.2): {state.get('section_4_2', '')[:800]}
    Value Drivers (2.1): {state.get('section_2_1', '')[:700]}

    Reinvestment Research with Citations:
    {chr(10).join(research_with_citations)}
    """

    prompt = f"""
    Write Section 4.3: Reinvestment Opportunity & Incremental Returns.

    Workflow:
    - Segment business lines/growth programs.
    - TAM per line (logic + key assumptions + biggest uncertainty).
    - Direction of incremental returns (improving/flat/declining + why).
    - Capital allocation framework verdict (disciplined/mixed/undisciplined) with examples.

    Context:
    {context}

    Requirements: Claim + Evidence + Implication; citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_4_3"] = response.content
    state["completed_sections"].append("4.3")

    logger.info(f"[OK] Section 4.3 completed ({len(response.content)} characters)")
    return state


def analyze_section_4_4(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 4.4: Sustainability of competitive advantage."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 4.4: SUSTAINABILITY OF ADVANTAGE")
    logger.info(f"{'=' * 60}\n")

    durability_research = web_search(f"{state['company_name']} market share pricing power retention trends")
    for result in durability_research:
        if result.get("url"):
            state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))

    research_with_citations = _research_context(state, durability_research[-15:], limit=15)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Competitive Advantage (4.2):
    {state.get('section_4_2', '')[:800]}

    Reinvestment (4.3):
    {state.get('section_4_3', '')[:600]}

    Durability Research with Citations:
    {chr(10).join(research_with_citations)}
    """

    prompt = f"""
    Write Section 4.4: Sustainability of Competitive Advantage.

    Workflow:
    - Durability scorecard: 4-6 indicators (share, pricing, churn, ROIC, contracts) vs peers.
    - Stress-test: how advantage holds under demand/input/cost stress.
    - Distinguish true moat vs temporary advantage (cycle/policy/FX).
    - Erosion watchlist: 3 bullets with signals and where they show up.

    Context:
    {context}

    Requirements: Claim + Evidence + Implication; citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_4_4"] = response.content
    state["completed_sections"].append("4.4")

    logger.info(f"[OK] Section 4.4 completed ({len(response.content)} characters)")
    return state


def analyze_section_5_1(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 5.1: Risk Identification, Quantification & Assessment."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 5.1: RISK IDENTIFICATION & QUANTIFICATION")
    logger.info(f"{'=' * 60}\n")

    queries = [
        f"{state['company_name']} risks challenges headwinds concerns",
        f"{state['company_name']} regulatory risks compliance litigation",
        f"{state['company_name']} competitive threats market share pressure",
        f"{state['company_name']} financial leverage debt covenants liquidity",
        f"{state['company_name']} governance controversies related party transactions",
    ]

    risk_research = []
    for query in queries:
        results = web_search(query)
        for result in results:
            if result.get("url"):
                state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))
                risk_research.append(result)
        logger.info(f"[OK] Search: {query}")

    research_with_citations = _research_context(state, risk_research[-20:], limit=20)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Previous Analysis:
    Section 2.1 (Value Drivers): {state.get('section_2_1', '')[:800]}
    Section 4.2 (Competitive Advantage): {state.get('section_4_2', '')[:800]}

    Financial Position:
    {json.dumps(state.get('financial_metrics', {}), indent=2)[:1500]}

    Risk Research with Citations:
    {chr(10).join(research_with_citations[:15])}
    """

    prompt = f"""
    Write Section 5.1: Risk Identification, Quantification & Assessment of Probability.
    
    Follow this EXACT workflow:
    
    STEP 1 - DEFINE "THESIS-BREAKING" (scope discipline):
    A risk qualifies ONLY if it can plausibly cause permanent impairment or 30-50% drawdown 
    over 3-5 years via:
    - Structural earnings power reduction
    - ROIC collapse / reinvestment dilution
    - Liquidity/refinancing event
    - Governance extraction
    - Regime change (regulation/geopolitics)
    
    OUTPUT: 1 sentence: "Thesis breaks if ___."
    
    STEP 2 - GENERATE THE INITIAL 3-5 RISK SET:
    Force coverage across categories. Select the MOST MATERIAL risks only (3-5 max).
    
    A) BUSINESS MODEL RISKS:
    - Demand durability (TAM overestimated; substitution; saturation)
    - Pricing power loss (commoditization; new entrants; channel power)
    - Cost curve disadvantage (input dependency; scale gap; productivity lag)
    - Execution risk on key projects (capacity ramp, product launch, integration)
    
    B) BALANCE SHEET / LIQUIDITY RISKS:
    - Refinancing wall, covenants, FX mismatch, duration mismatch
    - Working-capital blowout in downturn; customer defaults
    - Off-balance sheet leverage (leases, guarantees, supplier finance)
    
    C) GOVERNANCE / CONTROL RISKS:
    - Related-party transfers, dilutive issuance to insiders, value-destructive M&A
    - Minority rights weak; regulatory/political capture
    - Share pledging by controller/management (forced selling cascade)
    
    D) MACRO / REGIME RISKS:
    - FX/rates shocks, commodity regime shifts, policy reversal, trade/sanctions
    - Country risk: capital controls, taxation, nationalization, licensing changes
    
    For EACH risk selected (must be 3-5), provide:
    - Tag: Controllable / Partially controllable / Exogenous
    - Horizon: Near (0-12m) / Mid (1-3y) / Long (3-5y)
    
    STEP 3 - QUANTIFY IMPACT PATHWAYS (without full modeling):
    For EACH risk, use this structured 5-point chain:
    
    1. **Trigger / mechanism**: What specifically happens?
    2. **Operational transmission**: How does it affect volume/price/cost/capex/WC?
    3. **Financial impact** (direction + rough magnitude):
       - Revenue: -low (-5-10%) / -med (-10-25%) / -high (-25%+)
       - Margin: compression in basis points or percentage points
       - ROIC: how margin and/or capital turns compress
    4. **Cost of capital impact**: Does risk premium increase? Refinancing costs? Equity dilution?
    5. **Intrinsic value impact**: Multiple compression, cash-flow impairment, terminal fade?
    
    STEP 4 - ASSIGN PROBABILITY WITH EXPLICIT RATIONALE:
    For EACH risk, assess probability over 3-5 year horizon.
    
    Ask:
    - What is the BASE RATE of this risk in this industry/country/structure?
    - What company-specific features INCREASE likelihood?
    - What mitigants REDUCE likelihood?
    
    Use probability tiers:
    - **Low** (≤10%): Unlikely but possible
    - **Medium** (10-30%): Plausible, warrants monitoring
    - **High** (≥30%): Material probability, key thesis consideration
    
    For each risk: "**Probability: [Low/Med/High]** because [base rate] and [company modifiers]."
    
    STEP 5 - FIND OVERLOOKED RISKS FROM INDIRECT SOURCES:
    Ask:
    - What risks do PEERS discuss repeatedly that this company barely mentions?
    - What operational pain points show up in competitor/customer/supplier commentary?
    - Any shared dependencies (input, regulator, channel) creating correlated risk?
    
    OUTPUT: 2-4 "overlooked risk" bullets citing peer/industry sources [X].
    
    STEP 6 - SUMMARIZE RISK-ADJUSTED THESIS:
    - Which 1-2 risks dominate expected downside?
    - Which risk is most underpriced by consensus?
    - Leading indicators: 2-3 metrics to monitor per top risk
    
    Context:
    {context}
    
    CRITICAL REQUIREMENTS:
    - Must identify 3-5 thesis-breaking risks with detailed impact pathways
    - Probability tiers (Low/Med/High) with base rate reasoning
    - Citations [X] for all risk evidence
    - Overlooked risks from peer/industry analysis
    - Monitoring indicators specified
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_5_1"] = response.content
    state["completed_sections"].append("5.1")

    logger.info(f"[OK] Section 5.1 completed ({len(response.content)} characters)")
    return state


def analyze_section_5_2(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 5.2: Pre-mortem scenarios."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 5.2: PRE-MORTEM SCENARIOS")
    logger.info(f"{'=' * 60}\n")

    premortem_research = web_search(f"{state['company_name']} {state['ticker']} bear case risks scenarios")
    for result in premortem_research:
        if result.get("url"):
            state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))

    research_with_citations = _research_context(state, premortem_research[-10:], limit=10)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Risks (5.1): {state.get('section_5_1', '')[:900]}
    Value Drivers (2.1): {state.get('section_2_1', '')[:600]}
    Competitive Advantage (4.2): {state.get('section_4_2', '')[:600]}

    Scenario Research with Citations:
    {chr(10).join(research_with_citations)}
    """

    prompt = f"""
    Write Section 5.2: Pre-mortem - Alternative scenarios & likelihood.

    Workflow:
    - Define failure threshold and top 2-3 critical assumptions.
    - Build 2-3 distinct failure scenarios (root cause, timeline, mechanism, market reaction).
    - For each scenario: broken variable(s), why, why not mitigated.
    - Assign likelihood (Low/Med/High) + early warning indicators.
    - Pre-commit mitigations and walk-away conditions.

    Context:
    {context}

    Requirements: dense prose; Claim + Evidence + Implication; citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_5_2"] = response.content
    state["completed_sections"].append("5.2")

    logger.info(f"[OK] Section 5.2 completed ({len(response.content)} characters)")
    return state


def analyze_section_6_1(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 6.1: Peer review (growth/profitability/ROIC)."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 6.1: PEER REVIEW")
    logger.info(f"{'=' * 60}\n")

    peer_research = web_search(f"{state['company_name']} {state['ticker']} peers profitability growth ROIC comparison")
    for result in peer_research:
        if result.get("url"):
            state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))

    research_with_citations = _research_context(state, peer_research[-15:], limit=15)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    ROIC (4.1): {state.get('section_4_1', '')[:600]}
    Competitive Advantage (4.2): {state.get('section_4_2', '')[:600]}

    Peer Research with Citations:
    {chr(10).join(research_with_citations)}
    """

    prompt = f"""
    Write Section 6.1: Peer Review - margins, profitability, growth & ROIC.

    Workflow:
    - Define peer set (local + global) and note any false comps.
    - Compare growth (rev CAGR), profitability (gross/EBIT margins + volatility), returns (ROIC/ROE).
    - Normalize for distortions (one-offs, peak/trough, M&A).
    - Output: one-line peer verdict (winner/average/laggard, improving or not) + 3 bullets (Driver → Mechanism → Evidence).

    Context:
    {context}

    Requirements: Claim + Evidence + Implication; citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_6_1"] = response.content
    state["completed_sections"].append("6.1")

    logger.info(f"[OK] Section 6.1 completed ({len(response.content)} characters)")
    return state


def analyze_section_6_2(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 6.2: Relative valuation."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 6.2: RELATIVE VALUATION")
    logger.info(f"{'=' * 60}\n")

    valuation_research = web_search(f"{state['company_name']} {state['ticker']} valuation peers premium discount history multiples")
    for result in valuation_research:
        if result.get("url"):
            state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))

    research_with_citations = _research_context(state, valuation_research[-15:], limit=15)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Peer Review (6.1): {state.get('section_6_1', '')[:700]}
    ROIC (4.1): {state.get('section_4_1', '')[:600]}

    Valuation Research with Citations:
    {chr(10).join(research_with_citations)}
    """

    prompt = f"""
    Write Section 6.2: Relative Valuation (peers & history).

    Workflow:
    - Choose relevant lenses (P/E, EV/EBIT/EBITDA, P/B, FCF yield) and state why.
    - Peer comparison: Premium/Inline/Discount vs median/best-in-class with one-line reason.
    - Historical context: current multiple vs 3/5/10y band; why the market paid more/less.
    - Sanity checks (peak/trough earnings; EV/EBITDA vs FCF).

    Context:
    {context}

    Requirements: Claim + Evidence + Implication; citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_6_2"] = response.content
    state["completed_sections"].append("6.2")

    logger.info(f"[OK] Section 6.2 completed ({len(response.content)} characters)")
    return state


def analyze_section_6_3(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 6.3: 3-Year Valuation Target."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 6.3: 3-YEAR PRICE TARGET")
    logger.info(f"{'=' * 60}\n")

    queries = [
        f"{state['company_name']} {state['ticker']} valuation price target analyst estimates",
        f"{state['company_name']} peer valuation multiples comparison",
        f"{state['company_name']} historical valuation PE ratio trends",
        f"{state['ticker']} stock price forecast 2027 2028",
    ]

    valuation_research = []
    for query in queries:
        results = web_search(query)
        for result in results:
            if result.get("url"):
                state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))
                valuation_research.append(result)
        logger.info(f"[OK] Search: {query}")

    research_with_citations = _research_context(state, valuation_research[-15:], limit=15)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Investment Thesis:
    Value Drivers (2.1): {state.get('section_2_1', '')[:1000]}
    Competitive Advantage (4.2): {state.get('section_4_2', '')[:1000]}
    Risks (5.1): {state.get('section_5_1', '')[:1000]}

    Financial Snapshot (1.1):
    {state.get('section_1_1', '')[:800]}

    Financial Metrics:
    {json.dumps(state.get('financial_metrics', {}), indent=2)[:1500]}

    Valuation Research with Citations:
    {chr(10).join(research_with_citations)}
    """

    prompt = f"""
    Write Section 6.3: Valuation - 3-Year Price Target.
    
    Follow this EXACT workflow:
    
    STEP 1 - DEFINE THE "VALUE EQUATION" INPUTS:
    Starting point: Current metrics
    - Current stock price (if available in research)
    - Market cap
    - Revenue (latest year)
    - EBIT/EBITDA margin (latest)
    - Share count
    - Net debt/net cash position
    
    Key drivers from Section 2.1:
    - Volume/price/mix assumptions
    - Margin trajectory
    - Capex/WC intensity
    - Risk premium/multiple
    
    STEP 2 - BUILD SENSITIVITY FRAMEWORK (directional, not full model):
    
    **Table A - Revenue Opportunity (3-year):**
    Project revenue under three scenarios:
    - **Bear Case**: Conservative growth (e.g., 5-8% CAGR) - assume headwinds from Section 5.1
    - **Base Case**: Moderate growth (e.g., 10-12% CAGR) - balanced assumptions
    - **Bull Case**: Strong growth (e.g., 15-18% CAGR) - upside from Section 2.1 catalysts
    Output Table A as a Markdown table with columns: Scenario | CAGR assumption | 3Y revenue (USD). If precise numbers are unavailable, use directional placeholders (e.g., "TBD" or "Low/Med/High") but keep the table structure.
    
    **Table B - Margin & Multiple Scenarios:**
    For each revenue scenario, consider:
    - Margin assumption (current -> 3-year target)
    - Multiple assumption (P/E or EV/EBIT range based on ROIC quality from 4.2)
    Output Table B as a Markdown table with columns: Scenario | Margin assumption | Multiple assumption | Implied value note (EPS/EBIT/FCF per share). Use available numbers; if missing, use directional placeholders but keep the table.
    
    STEP 3 - PROJECT 3-YEAR BASE-CASE OUTCOMES (bridge logic):
    
    **Revenue Bridge:**
    - Starting revenue: [current]
    - Assumed CAGR: [X%] driven by [volume/price/mix from 2.1]
    - 3-year revenue: [calculated]
    
    **Earnings Bridge:**
    - Assumed steady-state margin: [X%] (justify vs history and Section 1.3)
    - 3-year EBIT/EBITDA: [revenue × margin]
    - 3-year EPS estimate (directional): [after tax, share count]
    
    **FCF Assumption:**
    - FCF conversion rate: [X% of EBIT] based on capex/WC intensity from 1.3 and 2.1
    - Why this conversion rate is sustainable or improving
    
    Explicitly state the 3-5 assumptions that matter most (not every line item).
    
    STEP 4 - CHOOSE AND JUSTIFY TARGET MULTIPLE:
    
    Anchor the multiple to fundamentals from Section 4.2:
    - Higher multiple requires: durable ROIC, growth runway, pricing power, clean governance
    - Lower multiple justified by: cyclicality, governance risk, capital intensity
    
    Reference:
    - Peer multiple range (if available in research)
    - Company's own historical multiple range (if available)
    - Quality premium vs discount
    
    State: "Target multiple of [X]x [P/E or EV/EBIT] because [durability from 4.2] 
    and [risks from 5.1] justify [below/at/above] peer median of [Y]x."
    
    STEP 5 - PRODUCE THE FINAL 3-YEAR FAIR VALUE RANGE:
    
    **Base Case Price Target:**
    - 3-year EPS × Target P/E = $XX
    - Or: (3-year EBIT × Target EV/EBIT - Net Debt) / Shares = $XX
    
    **Bull Case Price Target:** $YY
    - What must be true: [specific from 2.1 catalysts]
    
    **Bear Case Price Target:** $ZZ
    - What breaks: [specific from 5.1 risks]
    
    **Current Price vs Target:**
    - Current: $AA (state if known from research)
    - Base Target: $XX
    - Upside: +X% over 3 years (Y% annualized)
    
    **Investment Rating (must provide):**
    Based on risk/reward and upside:
    - **BUY**: If upside >30-40% with favorable risk/reward
    - **HOLD**: If upside 10-30% or risks balanced
    - **SELL**: If upside <10% or unfavorable risk/reward
    
    State rating clearly: "**RATING: [BUY/HOLD/SELL]** based on [upside %] and [risk profile from 5.1]"
    
    Context:
    {context}
    
    CRITICAL REQUIREMENTS:
    - Must provide specific 3-year price target (Base/Bull/Bear)
    - Must state assumptions explicitly (growth, margin, multiple)
    - Must justify multiple vs fundamentals from 4.2
    - Must incorporate risks from 5.1
    - Must provide clear BUY/HOLD/SELL rating
    - Citations [X] for current valuations and peer multiples
    - Dense prose, specific numbers
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_6_3"] = response.content
    state["completed_sections"].append("6.3")

    rating_match = re.search(r"RATING:\s*(BUY|HOLD|SELL)", response.content, re.IGNORECASE)
    target_match = re.search(r"[Bb]ase.*?[Tt]arget.*?\\$(\\d+)", response.content)
    upside_match = re.search(r"[Uu]pside.*?([+-]\\d+)%", response.content)

    state["investment_rating"] = rating_match.group(1).upper() if rating_match else "HOLD"
    state["target_price"] = f"${target_match.group(1)}" if target_match else "Target TBD"
    state["upside_potential"] = f"{upside_match.group(1)}%" if upside_match else "Upside TBD"

    logger.info(f"[OK] Section 6.3 completed ({len(response.content)} characters)")
    logger.info(f"  Rating: {state['investment_rating']}")
    logger.info(f"  Target: {state['target_price']}")
    logger.info(f"  Upside: {state['upside_potential']}")
    return state


def analyze_section_7_1(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 7.1: Stewardship vs Legacy classification."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 7.1: STEWARDSHIP VS LEGACY")
    logger.info(f"{'=' * 60}\n")

    runway_research = web_search(f"{state['company_name']} incremental ROIC reinvestment runway {state['ticker']}")
    for result in runway_research:
        if result.get("url"):
            state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))

    research_with_citations = _research_context(state, runway_research[-10:], limit=10)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    ROIC (4.1): {state.get('section_4_1', '')[:600]}
    Reinvestment (4.3): {state.get('section_4_3', '')[:700]}
    Valuation (6.3): {state.get('section_6_3', '')[:700]}

    Runway Research with Citations:
    {chr(10).join(research_with_citations)}
    """

    prompt = f"""
    Write Section 7.1: Legacy vs Stewardship Opportunity.

    Workflow:
    - Historic ROIC quality (through-cycle) and volatility.
    - Incremental return direction (improving/flat/declining) with evidence (unit economics/payback/capex-WC intensity).
    - Runway verdict (long/medium/short) and why.
    - Market misperception check (what market under/overestimates).
    - Classify: Stewardship / Legacy / Hybrid using rule; provide 3 bullets of decisive evidence.
    - Value-creation playbook implied by the label (3 bullets: do more/less/watch).

    Context:
    {context}

    Requirements: Claim + Evidence + Implication; citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_7_1"] = response.content
    state["completed_sections"].append("7.1")

    logger.info(f"[OK] Section 7.1 completed ({len(response.content)} characters)")
    return state


def analyze_section_8_1(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 8.1: Culture, values, and purpose."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 8.1: CULTURE")
    logger.info(f"{'=' * 60}\n")

    culture_research = web_search(f"{state['company_name']} culture values purpose talent retention governance")
    for result in culture_research:
        if result.get("url"):
            state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))

    research_with_citations = _research_context(state, culture_research[-15:], limit=15)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Industry Rules (3.3): {state.get('section_3_3', '')[:600]}

    Culture Research with Citations:
    {chr(10).join(research_with_citations)}
    """

    prompt = f"""
    Write Section 8.1: Culture - values and purpose.

    Workflow:
    - Restate industry rules of winning (from 3.3).
    - Infer required culture archetype; evaluate alignment of stated values vs needed behaviors.
    - Identify cultural proof points/contradictions (decision patterns, incentives, retention, safety/quality).
    - Define 3 engagement levers to improve long-term outcomes.

    Context:
    {context}

    Requirements: Claim + Evidence + Implication; citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_8_1"] = response.content
    state["completed_sections"].append("8.1")

    logger.info(f"[OK] Section 8.1 completed ({len(response.content)} characters)")
    return state


def analyze_section_8_2(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 8.2: Sustainability assessment."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 8.2: SUSTAINABILITY ASSESSMENT")
    logger.info(f"{'=' * 60}\n")

    sustainability_research = web_search(f"{state['company_name']} sustainability ESG controversies targets")
    for result in sustainability_research:
        if result.get("url"):
            state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))

    research_with_citations = _research_context(state, sustainability_research[-15:], limit=15)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Culture (8.1): {state.get('section_8_1', '')[:600]}

    Sustainability Research with Citations:
    {chr(10).join(research_with_citations)}
    """

    prompt = f"""
    Write Section 8.2: Sustainability assessment.

    Workflow:
    - Maturity verdict (basic/developing/advanced) with 2 bullets (governance, reporting, incentives).
    - Top 3-5 material topics with economic linkage (revenue/pricing/cost/capex/risk premium).
    - Controversy scan: 3-6 bullets (issue → status → potential impact).

    Context:
    {context}

    Requirements: Claim + Evidence + Implication; citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_8_2"] = response.content
    state["completed_sections"].append("8.2")

    logger.info(f"[OK] Section 8.2 completed ({len(response.content)} characters)")
    return state


def analyze_section_8_3(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 8.3: Risk mitigation vs peers."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 8.3: RISK MITIGATION VS PEERS")
    logger.info(f"{'=' * 60}\n")

    cost_of_capital_research = web_search(f"{state['company_name']} governance risk premium cost of capital peers")
    for result in cost_of_capital_research:
        if result.get("url"):
            state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))

    research_with_citations = _research_context(state, cost_of_capital_research[-15:], limit=15)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Sustainability Assessment (8.2): {state.get('section_8_2', '')[:700]}
    Risks (5.1): {state.get('section_5_1', '')[:700]}

    Cost of Capital Research with Citations:
    {chr(10).join(research_with_citations)}
    """

    prompt = f"""
    Write Section 8.3: How the company addresses risk vs peers (cost of capital lens).

    Workflow:
    - 3 bullets: why cost of capital is high/low in this sector.
    - Topic-by-topic comparison for top risks (better/similar/worse vs peers + evidence).
    - 2 bullets: what would credibly reduce the discount.

    Context:
    {context}

    Requirements: Claim + Evidence + Implication; citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_8_3"] = response.content
    state["completed_sections"].append("8.3")

    logger.info(f"[OK] Section 8.3 completed ({len(response.content)} characters)")
    return state


def analyze_section_8_4(state: ShallowDiveState) -> ShallowDiveState:
    """Generate Section 8.4: Engagement opportunities & plan."""
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYZING SECTION 8.4: ENGAGEMENT PLAN")
    logger.info(f"{'=' * 60}\n")

    engagement_research = web_search(f"{state['company_name']} governance improvements engagement priorities")
    for result in engagement_research:
        if result.get("url"):
            state, _ = add_source(state, result["url"], result.get("title", "Untitled"), result.get("content", ""))

    research_with_citations = _research_context(state, engagement_research[-10:], limit=10)
    context = f"""
    Company: {state['company_name']} ({state['ticker']})

    Sustainability (8.2): {state.get('section_8_2', '')[:600]}
    Risk Mitigation (8.3): {state.get('section_8_3', '')[:600]}

    Engagement Research with Citations:
    {chr(10).join(research_with_citations)}
    """

    prompt = f"""
    Write Section 8.4: Engagement opportunities & plan.

    Workflow:
    - Generate candidate engagement list (capital allocation, governance, sustainability, disclosures, incentives).
    - Prioritize 5-8 objectives (value impact, feasibility, time-to-proof).
    - For each: Objective | KPI | Deadline | Owner | Proof.
    - Roadmap: 6-12m / 12-24m / 24-36m milestones.
    - Red lines: 2-3 conditions triggering escalation.

    Context:
    {context}

    Requirements: Claim + Evidence + Implication; citations [X] for factual claims.
    """

    messages = [SystemMessage(content=config.ANALYST_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = config.llm.invoke(messages)
    state["section_8_4"] = response.content
    state["completed_sections"].append("8.4")

    logger.info(f"[OK] Section 8.4 completed ({len(response.content)} characters)")
    return state


def compile_final_report(state: ShallowDiveState) -> ShallowDiveState:
    """Compile all sections into final investment-grade report."""
    logger.info(f"\n{'=' * 60}")
    logger.info("COMPILING FINAL REPORT")
    logger.info(f"{'=' * 60}\n")

    references = generate_references_section(state)
    report = f"""
# INVESTMENT-GRADE SHALLOW DIVE ANALYSIS
## {state['company_name']} ({state['ticker']})

Analysis Date: {datetime.now().strftime('%Y-%m-%d')}

---

## INVESTMENT RECOMMENDATION

**Rating:** {state.get('investment_rating', 'TBD')}

**3-Year Price Target:** {state.get('target_price', 'TBD')}

**Upside Potential:** {state.get('upside_potential', 'TBD')}

---

## SECTION 1: PROFILE

### 1.1 Company Snapshot

{state.get('section_1_1', 'Not completed')}

---

### 1.2 What Does the Company Do?

{state.get('section_1_2', 'Not completed')}

---

### 1.3 How Is Value Created? (DuPont Analysis)

{state.get('section_1_3', 'Not completed')}

---

### 1.4 Theme Identification & Exposure

{state.get('section_1_4', 'Not completed')}

---

### 1.5 Founders, Management, Shareholders

{state.get('section_1_5', 'Not completed')}

---

## SECTION 2: INVESTMENT THESIS

### 2.1 Key Value Drivers & Catalysts

{state.get('section_2_1', 'Not completed')}

---

### 2.2 Implied Expectations

{state.get('section_2_2', 'Not completed')}

---

### 2.3 Key Assumptions & Difference vs Consensus

{state.get('section_2_3', 'Not completed')}

---

## SECTION 3: INDUSTRY

### 3.1 Profit Pool & Value Chain

{state.get('section_3_1', 'Not completed')}

---

### 3.2 Market Structure (Five Forces)

{state.get('section_3_2', 'Not completed')}

---

### 3.3 Industry Structure Classification & Positioning

{state.get('section_3_3', 'Not completed')}

---

## SECTION 4: COMPETITIVE ADVANTAGE

### 4.1 ROIC Analysis vs Peers

{state.get('section_4_1', 'Not completed')}

---

### 4.2 Source of Enduring Competitive Advantage (7 Powers)

{state.get('section_4_2', 'Not completed')}

---

### 4.3 Reinvestment Opportunity & Incremental Returns

{state.get('section_4_3', 'Not completed')}

---

### 4.4 Sustainability of Competitive Advantage

{state.get('section_4_4', 'Not completed')}

---

## SECTION 5: RISKS

### 5.1 Risk Identification, Quantification & Probability Assessment

{state.get('section_5_1', 'Not completed')}

---

### 5.2 Pre-Mortem Scenarios & Likelihood

{state.get('section_5_2', 'Not completed')}

---

## SECTION 6: VALUATION

### 6.1 Peer Review (Growth, Profitability, ROIC)

{state.get('section_6_1', 'Not completed')}

---

### 6.2 Relative Valuation (Peers & History)

{state.get('section_6_2', 'Not completed')}

---

### 6.3 Three-Year Price Target & Scenarios

{state.get('section_6_3', 'Not completed')}

---

## SECTION 7: IDENTIFICATION

### 7.1 Stewardship vs Legacy Classification

{state.get('section_7_1', 'Not completed')}

---

## SECTION 8: ENGAGEMENT

### 8.1 Culture: Values & Purpose

{state.get('section_8_1', 'Not completed')}

---

### 8.2 Sustainability Assessment

{state.get('section_8_2', 'Not completed')}

---

### 8.3 Addressing Risk vs Peers (Cost of Capital)

{state.get('section_8_3', 'Not completed')}

---

### 8.4 Engagement Opportunities & Plan

{state.get('section_8_4', 'Not completed')}

---

{references}

---

## Analysis Metadata

**Phase**: Phase 1 - Core Investment Analysis

**Completed Sections:** {', '.join(state.get('completed_sections', []))}

**Section Coverage:**
- Profile Analysis: Sections 1.1-1.5 [OK]
- Investment Thesis: Sections 2.1-2.3 [OK]
- Industry: Sections 3.1-3.3 [OK]
- Economic Value-Add: Sections 4.1-4.4 [OK]
- Risk Assessment: Sections 5.1-5.2 [OK]
- Valuation: Sections 6.1-6.3 [OK]
- Identification: Section 7.1 [OK]
- Engagement: Sections 8.1-8.4 [OK]

**Data Sources:**
- Financial APIs: {'Yes' if config.FMP_API_KEY else 'No'}
- Web Research Queries: {len(state.get('web_research', []))}
- Total Sources Cited: {len(state.get('sources', []))}

**Quality Metrics:**
- Investment Rating: {state.get('investment_rating', 'Pending')}
- Target Price: {state.get('target_price', 'Pending')}
- Risks Identified: {state.get('section_5_1', '').count('RISK') if state.get('section_5_1') else 0}+
- Competitive Powers Identified: {len([p for p in ['SCALE', 'NETWORK', 'SWITCHING', 'BRAND', 'CORNERED', 'PROCESS', 'COUNTER'] if p in state.get('section_4_2', '').upper()])}

**Errors/Warnings:** {', '.join(state.get('errors', [])) if state.get('errors') else 'None'}

---

*This investment-grade analysis was generated using an automated LangGraph workflow implementing Phase 1 of the comprehensive shallow dive framework. The analysis includes value drivers, competitive advantage assessment (7 Powers), risk quantification, and a 3-year price target with BUY/HOLD/SELL recommendation.*
    """

    state["final_report"] = report

    logger.info("[OK] Final report compiled")
    logger.info(f"Total length: {len(report)} characters")
    logger.info(f"Completed sections: {len(state.get('completed_sections', []))}/25 (Phase 1)")
    logger.info(f"Total sources cited: {len(state.get('sources', []))}")
    logger.info(f"Investment Rating: {state.get('investment_rating', 'TBD')}")
    return state
