"""
utils/helpers.py
Console rendering helpers using the 'rich' library.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from graph.state import SIFACTState

console = Console()

STANCE_COLORS = {
    "supported":    "green",
    "baseless":     "red",
    "inconclusive": "yellow",
}

VERDICT_STYLES = {
    "REAL":      ("bold green",  "✅ REAL"),
    "FAKE":      ("bold red",    "❌ FAKE"),
    "UNCERTAIN": ("bold yellow", "⚠️  UNCERTAIN"),
}


def print_results(state: SIFACTState) -> None:
    """Pretty-print the full pipeline output to the console."""

    # ── Claims table ──────────────────────────────────────────────────────────
    claims_table = Table(title="Phase 1 – Extracted Claims", box=box.ROUNDED, show_lines=True)
    claims_table.add_column("ID",   style="dim", width=14)
    claims_table.add_column("Type", width=10)
    claims_table.add_column("Claim Text")

    for claim in state.get("claims", []):
        claims_table.add_row(
            claim["id"],
            f"[bold cyan]{claim['type']}[/]",
            claim["text"],
        )
    console.print(claims_table)

    # ── Stances table ─────────────────────────────────────────────────────────
    stances_table = Table(title="Phase 2 – Verification Results", box=box.ROUNDED, show_lines=True)
    stances_table.add_column("Claim ID",  style="dim", width=14)
    stances_table.add_column("Stance",    width=14)
    stances_table.add_column("Conf",      width=6)
    stances_table.add_column("Evidence Summary")

    for s in state.get("verified_stances", []):
        color = STANCE_COLORS.get(s["stance"], "white")
        stances_table.add_row(
            s["claim_id"],
            f"[{color}]{s['stance'].upper()}[/]",
            f"{s['confidence']:.2f}",
            s["evidence_summary"][:100],
        )
    console.print(stances_table)

    # ── Final verdict panel ───────────────────────────────────────────────────
    verdict = state.get("final_verdict", "UNCERTAIN")
    style, label = VERDICT_STYLES.get(verdict, ("bold white", verdict))
    conf = state.get("confidence_score", 0.0)

    panel_text = (
        f"[{style}]{label}[/]\n"
        f"Confidence: [bold]{conf:.0%}[/]\n\n"
        f"{state.get('explanation', '')}"
    )
    console.print(
        Panel(
            panel_text,
            title="Phase 3 – Final Verdict",
            border_style=style.split()[-1],
            expand=False,
        )
    )