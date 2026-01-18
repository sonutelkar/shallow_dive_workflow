"""Build and compile the LangGraph workflow."""

from langgraph.graph import StateGraph, END

from .sections import (
    analyze_section_1_1,
    analyze_section_1_2,
    analyze_section_1_3,
    analyze_section_1_4,
    analyze_section_1_5,
    analyze_section_2_1,
    analyze_section_2_2,
    analyze_section_2_3,
    analyze_section_3_1,
    analyze_section_3_2,
    analyze_section_3_3,
    analyze_section_4_1,
    analyze_section_4_2,
    analyze_section_4_3,
    analyze_section_4_4,
    analyze_section_5_1,
    analyze_section_5_2,
    analyze_section_6_1,
    analyze_section_6_2,
    analyze_section_6_3,
    analyze_section_7_1,
    analyze_section_8_1,
    analyze_section_8_2,
    analyze_section_8_3,
    analyze_section_8_4,
    compile_final_report,
    gather_financial_data,
    initialize_research,
)
from .state import ShallowDiveState


def build_workflow():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(ShallowDiveState)

    workflow.add_node("initialize", initialize_research)
    workflow.add_node("gather_financials", gather_financial_data)
    workflow.add_node("section_1_1", analyze_section_1_1)
    workflow.add_node("section_1_2", analyze_section_1_2)
    workflow.add_node("section_1_3", analyze_section_1_3)
    workflow.add_node("section_1_4", analyze_section_1_4)
    workflow.add_node("section_1_5", analyze_section_1_5)

    workflow.add_node("section_2_1", analyze_section_2_1)
    workflow.add_node("section_2_2", analyze_section_2_2)
    workflow.add_node("section_2_3", analyze_section_2_3)

    workflow.add_node("section_3_1", analyze_section_3_1)
    workflow.add_node("section_3_2", analyze_section_3_2)
    workflow.add_node("section_3_3", analyze_section_3_3)

    workflow.add_node("section_4_1", analyze_section_4_1)
    workflow.add_node("section_4_2", analyze_section_4_2)
    workflow.add_node("section_4_3", analyze_section_4_3)
    workflow.add_node("section_4_4", analyze_section_4_4)

    workflow.add_node("section_5_1", analyze_section_5_1)
    workflow.add_node("section_5_2", analyze_section_5_2)

    workflow.add_node("section_6_1", analyze_section_6_1)
    workflow.add_node("section_6_2", analyze_section_6_2)
    workflow.add_node("section_6_3", analyze_section_6_3)

    workflow.add_node("section_7_1", analyze_section_7_1)
    workflow.add_node("section_8_1", analyze_section_8_1)
    workflow.add_node("section_8_2", analyze_section_8_2)
    workflow.add_node("section_8_3", analyze_section_8_3)
    workflow.add_node("section_8_4", analyze_section_8_4)

    workflow.add_node("compile_report", compile_final_report)

    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "gather_financials")
    workflow.add_edge("gather_financials", "section_1_1")
    workflow.add_edge("section_1_1", "section_1_2")
    workflow.add_edge("section_1_2", "section_1_3")
    workflow.add_edge("section_1_3", "section_1_4")
    workflow.add_edge("section_1_4", "section_1_5")
    workflow.add_edge("section_1_5", "section_2_1")
    workflow.add_edge("section_2_1", "section_2_2")
    workflow.add_edge("section_2_2", "section_2_3")
    workflow.add_edge("section_2_3", "section_3_1")
    workflow.add_edge("section_3_1", "section_3_2")
    workflow.add_edge("section_3_2", "section_3_3")
    workflow.add_edge("section_3_3", "section_4_1")
    workflow.add_edge("section_4_1", "section_4_2")
    workflow.add_edge("section_4_2", "section_4_3")
    workflow.add_edge("section_4_3", "section_4_4")
    workflow.add_edge("section_4_4", "section_5_1")
    workflow.add_edge("section_5_1", "section_5_2")
    workflow.add_edge("section_5_2", "section_6_1")
    workflow.add_edge("section_6_1", "section_6_2")
    workflow.add_edge("section_6_2", "section_6_3")
    workflow.add_edge("section_6_3", "section_7_1")
    workflow.add_edge("section_7_1", "section_8_1")
    workflow.add_edge("section_8_1", "section_8_2")
    workflow.add_edge("section_8_2", "section_8_3")
    workflow.add_edge("section_8_3", "section_8_4")
    workflow.add_edge("section_8_4", "compile_report")
    workflow.add_edge("compile_report", END)

    return workflow.compile()
