"""
main.py
SIFACT – Structured Intelligent Fact-Checking Tool
CLI entry-point.

Usage:
  # From text
  python main.py --article "Paste your news article text here..."

  # From file
  python main.py --file path/to/article.txt

  # Quick demo with a built-in sample
  python main.py --demo
"""

from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.rule import Rule

from graph.state import SIFACTState
from graph.workflow import sifact_graph
from utils.helpers import print_results

console = Console()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Demo article ──────────────────────────────────────────────────────────────
DEMO_ARTICLE = """
Scientists at MIT announced yesterday that they have successfully developed a new battery
technology that can charge a standard electric vehicle in under 5 minutes, compared to
the current average of 30-60 minutes for fast chargers. The breakthrough uses a new
lithium-metal anode combined with a solid-state electrolyte, achieving an energy density
of 500 Wh/kg — nearly double today's best commercial batteries. The research, published
in the journal Nature Energy, was funded by the US Department of Energy with a $50 million
grant. Lead researcher Dr. Sarah Chen stated the technology could reach mass production
by 2027. The university has already filed 12 patents related to the discovery.
"""


# ── Runner ────────────────────────────────────────────────────────────────────

def run(article: str) -> SIFACTState:
    console.print(Rule("[bold blue]SIFACT — Fake News Detector[/]"))
    console.print(f"\n[dim]Article preview:[/] {article[:200].strip()}…\n")

    initial_state: SIFACTState = {
        "article": article,
        "claims": [],
        "verified_stances": [],
        "is_fake": False,
        "confidence_score": 0.0,
        "final_verdict": "UNCERTAIN",
        "explanation": "",
        "error": None,
    }

    final_state: SIFACTState = sifact_graph.invoke(initial_state)

    console.print()
    print_results(final_state)
    console.print(Rule())

    return final_state


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SIFACT Fake News Detector")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--article", type=str, help="Article text (inline)")
    group.add_argument("--file",    type=str, help="Path to .txt file with the article")
    group.add_argument("--demo",    action="store_true", help="Run with built-in demo article")
    args = parser.parse_args()

    if args.demo:
        article = DEMO_ARTICLE
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            article = fh.read()
    elif args.article:
        article = args.article
    else:
        parser.print_help()
        sys.exit(1)

    article = article.strip()
    if not article:
        console.print("[red]Error:[/] Article text is empty.")
        sys.exit(1)

    final_state = run(article)

    # Machine-readable exit code: 1 = fake, 0 = real/uncertain
    sys.exit(1 if final_state.get("is_fake") else 0)


if __name__ == "__main__":
    main()