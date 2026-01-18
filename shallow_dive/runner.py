"""Execution helpers for running single or batch analyses."""

import os
import json
from datetime import datetime
from typing import Dict

import pandas as pd

from . import config
from .workflow import build_workflow

logger = config.logger


def analyze_single_company(company_name: str, ticker: str, output_dir: str = ".") -> Dict:
    """Analyze a single company and save the report."""
    initial_state = {
        "company_name": company_name,
        "ticker": ticker,
        "company_overview": {},
        "financial_metrics": {},
        "ownership_data": {},
        "market_data": {},
        "web_research": [],
        "sources": [],
        "source_map": {},
        "section_1_1": "",
        "section_1_2": "",
        "section_1_3": "",
        "section_1_4": "",
        "section_1_5": "",
        "section_2_1": "",
        "section_2_2": "",
        "section_2_3": "",
        "section_3_1": "",
        "section_3_2": "",
        "section_3_3": "",
        "section_4_1": "",
        "section_4_2": "",
        "section_4_3": "",
        "section_4_4": "",
        "section_5_1": "",
        "section_5_2": "",
        "section_6_1": "",
        "section_6_2": "",
        "section_6_3": "",
        "section_7_1": "",
        "section_8_1": "",
        "section_8_2": "",
        "section_8_3": "",
        "section_8_4": "",
        "investment_rating": "",
        "target_price": "",
        "upside_potential": "",
        "current_section": "",
        "completed_sections": [],
        "errors": [],
        "final_report": "",
    }

    logger.info(f"\n{'=' * 70}")
    logger.info("STARTING SHALLOW DIVE ANALYSIS")
    logger.info(f"Company: {company_name} ({ticker})")
    logger.info(f"{'=' * 70}\n")

    app = build_workflow()

    try:
        final_state = app.invoke(initial_state)

        logger.info(f"\n{'=' * 70}")
        logger.info("ANALYSIS COMPLETE")
        logger.info(f"{'=' * 70}\n")

        os.makedirs(output_dir, exist_ok=True)
        report_filename = os.path.join(output_dir, f"shallow_dive_{ticker}_{datetime.now().strftime('%Y%m%d')}.md")

        with open(report_filename, "w", encoding="utf-8") as file:
            file.write(final_state["final_report"])

        logger.info(f"[OK] Report saved to: {report_filename}")
        logger.info(f"Completed sections: {', '.join(final_state['completed_sections'])}")
        logger.info(f"Total sources cited: {len(final_state.get('sources', []))}")

        return {
            "company": company_name,
            "ticker": ticker,
            "status": "Success",
            "filename": report_filename,
            "sections": len(final_state["completed_sections"]),
            "sources": len(final_state.get("sources", [])),
        }

    except Exception as exc:  # pragma: no cover - runtime logging
        logger.exception(f"Error during execution: {exc}")
        return {"company": company_name, "ticker": ticker, "status": "Failed", "error": str(exc)}


def analyze_batch(companies_file: str, output_dir: str = "."):
    """Analyze multiple companies from a JSON file."""
    with open(companies_file, "r") as file:
        companies = json.load(file)

    results = []
    for company in companies:
        result = analyze_single_company(company["name"], company["ticker"], output_dir)
        results.append(result)

    return pd.DataFrame(results)


def main():
    """Main entry point for the script."""
    missing_keys = config.validate_api_keys()
    if missing_keys:
        logger.error(f"Missing required environment variables: {', '.join(missing_keys)}")
        return

    import argparse

    parser = argparse.ArgumentParser(
        description="Shallow Dive Company Analysis - Automated Equity Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single company analysis:
    python shallow_dive_workflow_phase1_complete.py --company "Alibaba Group" --ticker "BABA"
  
  Batch analysis:
    python shallow_dive_workflow_phase1_complete.py --batch companies.json
  
  Custom output directory:
    python shallow_dive_workflow_phase1_complete.py --company "Infosys" --ticker "INFY" --output-dir ./reports
        """,
    )

    parser.add_argument("--company", type=str, help="Company name")
    parser.add_argument("--ticker", type=str, help="Stock ticker symbol")
    parser.add_argument("--batch", type=str, help="Path to JSON file with companies list")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for reports (default: current directory)")

    args = parser.parse_args()

    if args.batch:
        if not os.path.exists(args.batch):
            logger.error(f"Batch file not found: {args.batch}")
            return

        logger.info(f"Running batch analysis from: {args.batch}")
        results_df = analyze_batch(args.batch, args.output_dir)

        summary_file = os.path.join(args.output_dir, f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        results_df.to_csv(summary_file, index=False, encoding="utf-8")

        logger.info(f"\n{'=' * 70}")
        logger.info("BATCH ANALYSIS COMPLETE")
        logger.info(f"{'=' * 70}")
        logger.info(f"\nResults summary saved to: {summary_file}")
        logger.info("\nSummary:")
        logger.info(results_df.to_string(index=False))

    elif args.company and args.ticker:
        analyze_single_company(args.company, args.ticker, args.output_dir)
    else:
        parser.print_help()
        logger.error("Either --company and --ticker, or --batch must be provided")


if __name__ == "__main__":
    main()
