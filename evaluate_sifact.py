"""
evaluate_sifact.py
==================
Evaluates the SIFACT pipeline against the AVeriTeC dataset (JSON format).

Usage
-----
# Evaluate on dev set (first 50 samples for a quick run):
    python evaluate_sifact.py --dataset dev.json --limit 50

# Full dev set evaluation with checkpointing:
    python evaluate_sifact.py --dataset dev.json --output results/

# Skip extraction agent — feed claims directly to verification (faster & fairer):
    python evaluate_sifact.py --dataset dev.json --direct-verify

# Resume an interrupted run:
    python evaluate_sifact.py --dataset dev.json --checkpoint results/checkpoint.json

Architecture Notes
------------------
SIFACT outputs: REAL | FAKE | UNCERTAIN
AVeriTeC labels: Supported | Refuted | Conflicting Evidence/Cherrypicking | Not Enough Evidence

Label Mapping (3-class, used for main metrics)
    REAL      →  Supported
    FAKE      →  Refuted
    UNCERTAIN →  Not Enough Evidence  OR  Conflicting Evidence/Cherrypicking
                 (SIFACT cannot distinguish these two — see per-class breakdown)

Key Differences vs. the FEVER/AVeriTeC paper baseline
    1. Retrieval: SIFACT uses live NewsAPI; paper uses static knowledge store
    2. Input:     SIFACT takes raw articles; this script adapts by passing claim text
    3. Labels:    SIFACT has 3 outputs; AVeriTeC has 4 — UNCERTAIN covers 2 of them
    4. Metrics:   Paper uses Hungarian-METEOR Q+A score; this script uses F1 + accuracy
                  (Hungarian-METEOR requires generated Q&A pairs that SIFACT doesn't produce)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import getpass
from pathlib import Path
from typing import Any

# ── Optional rich progress bar ────────────────────────────────────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[WARN] tqdm not installed — install with: pip install tqdm")

# ── Metrics ───────────────────────────────────────────────────────────────────
try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn not installed — install with: pip install scikit-learn")

# ── SIFACT imports (must be run from SIFACT repo root) ───────────────────────
try:
    from graph.state import SIFACTState
    from graph.workflow import sifact_graph
    from agents.verification_agent import _verify_single_claim
    from langchain_groq import ChatGroq
    from config.settings import GROQ_API_KEY1 as GROQ_API_KEY, VERIFICATION_MODEL
    from graph.state import Claim
    SIFACT_AVAILABLE = True
except ImportError as _e:
    SIFACT_AVAILABLE = False
    print(f"[ERROR] Could not import SIFACT modules: {_e}")
    print("Make sure you run this script from the root of the SIFACT repository.")
    sys.exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ── Constants ─────────────────────────────────────────────────────────────────

AVERITEC_LABELS = [
    "Supported",
    "Refuted",
    "Conflicting Evidence/Cherrypicking",
    "Not Enough Evidence",
]

AVERITEC_TO_3CLASS = {
    "Supported": "Supported",
    "Refuted": "Refuted",
    "Conflicting Evidence/Cherrypicking": "Uncertain",
    "Not Enough Evidence": "Uncertain",
}
LABELS_3CLASS = ["Supported", "Refuted", "Uncertain"]

SIFACT_TO_3CLASS = {
    "REAL": "Supported",
    "FAKE": "Refuted",
    "UNCERTAIN": "Uncertain",
}

API_DELAY_SECONDS = 2.5

# ── Rate-limit error detection keywords ───────────────────────────────────────
RATE_LIMIT_KEYWORDS = [
    "rate_limit_exceeded",
    "rate limit",
    "quota exceeded",
    "daily limit",
    "too many requests",
    "429",
    "tokens per day",
    "request limit",
]


# ── API Key Manager ───────────────────────────────────────────────────────────

class APIKeyManager:
    """
    Manages the active Groq API key.
    When a rate-limit / quota error is detected, prompts the user
    interactively for a new key and rebuilds the LLM client.
    """

    def __init__(self, initial_key: str, model: str):
        self.model = model
        self.current_key = initial_key
        self._llm = self._build_llm(initial_key)

    def _build_llm(self, key: str) -> ChatGroq:
        return ChatGroq(
            model=self.model,
            groq_api_key=key,
            temperature=0.0,
        )

    @property
    def llm(self) -> ChatGroq:
        return self._llm

    def is_rate_limit_error(self, exc: Exception) -> bool:
        """Return True if the exception looks like a rate-limit / quota error."""
        msg = str(exc).lower()
        return any(kw in msg for kw in RATE_LIMIT_KEYWORDS)

    def prompt_for_new_key(self) -> bool:
        """
        Interactively ask the user for a new API key.
        Returns True if a new key was provided, False if the user wants to abort.
        """
        print("\n" + "=" * 60)
        print("  ⚠  GROQ API RATE LIMIT / DAILY QUOTA REACHED")
        print("=" * 60)
        print("  The current API key has hit its daily token/request limit.")
        print("  Options:")
        print("    • Enter a new Groq API key to continue evaluation.")
        print("    • Press ENTER (blank) to stop and save progress.")
        print("=" * 60)

        try:
            # Use getpass so the key isn't echoed to the terminal
            new_key = getpass.getpass("  New Groq API key (input hidden): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Interrupted — saving checkpoint and exiting.")
            return False

        if not new_key:
            print("[INFO] No key provided — stopping evaluation.")
            return False

        self.current_key = new_key
        self._llm = self._build_llm(new_key)
        print("[INFO] API key updated successfully. Resuming evaluation…\n")
        return True

    def run_with_retry(self, fn, *args, max_key_rotations: int = 5, **kwargs):
        """
        Call fn(*args, **kwargs), automatically prompting for a new key
        whenever a rate-limit error is encountered.

        Parameters
        ----------
        fn               : callable that uses self.llm internally
        max_key_rotations: safety cap — won't prompt more than this many times
        """
        rotations = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if self.is_rate_limit_error(exc) and rotations < max_key_rotations:
                    logger.warning("Rate limit hit: %s", exc)
                    if self.prompt_for_new_key():
                        rotations += 1
                        # Small pause before retrying with the new key
                        time.sleep(2)
                        continue
                    else:
                        raise RuntimeError("Evaluation stopped by user after rate limit.") from exc
                raise  # Re-raise non-rate-limit errors immediately


# ── Global key manager (initialised in main) ─────────────────────────────────
key_manager: APIKeyManager | None = None


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "claims", "samples"):
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"Unexpected JSON structure in {path}.")


def extract_label(record: dict) -> str:
    label = record.get("label", "")
    for canonical in AVERITEC_LABELS:
        if label.strip().lower() == canonical.lower():
            return canonical
    logger.warning("Unrecognised label '%s' — treating as Not Enough Evidence", label)
    return "Not Enough Evidence"


# ── SIFACT pipeline wrappers ──────────────────────────────────────────────────

def run_full_pipeline(claim_text: str) -> dict:
    """Run the full 3-agent SIFACT pipeline."""
    initial_state: SIFACTState = {
        "article": claim_text,
        "claims": [],
        "verified_stances": [],
        "is_fake": False,
        "confidence_score": 0.0,
        "final_verdict": "UNCERTAIN",
        "explanation": "",
        "error": None,
    }
    return sifact_graph.invoke(initial_state)


def _direct_verify_inner(claim_text: str, claim_id: str, llm: ChatGroq) -> dict:
    """
    Core direct-verify logic, accepts an explicit llm instance so we can
    swap in a fresh client after a key rotation.
    """
    claim: Claim = Claim(id=claim_id, text=claim_text, type="central")

    try:
        stance = _verify_single_claim(claim, llm)
    except TypeError:
        stance = _verify_single_claim(claim)

    stance_to_verdict = {
        "supported":    "REAL",
        "baseless":     "FAKE",
        "inconclusive": "UNCERTAIN",
    }
    verdict = stance_to_verdict.get(stance["stance"], "UNCERTAIN")

    return {
        "final_verdict":     verdict,
        "confidence_score":  stance["confidence"],
        "explanation":       stance["evidence_summary"],
        "verified_stances":  [stance],
        "is_fake":           verdict == "FAKE",
    }


def run_direct_verify(claim_text: str, claim_id: str = "central") -> dict:
    """
    Bypass the extraction agent and feed the claim directly to the
    verification agent, routing through APIKeyManager for rate-limit handling.
    """
    assert key_manager is not None, "key_manager not initialised"

    def _call():
        return _direct_verify_inner(claim_text, claim_id, key_manager.llm)

    return key_manager.run_with_retry(_call)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"results": [], "processed_ids": []}


def save_checkpoint(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    gt_4class   = [r["ground_truth_label"] for r in results]
    pred_3class = [r["predicted_3class"] for r in results]
    gt_3class   = [AVERITEC_TO_3CLASS[g] for g in gt_4class]

    metrics: dict[str, Any] = {}

    if HAS_SKLEARN:
        metrics["accuracy_3class"] = round(accuracy_score(gt_3class, pred_3class), 4)
        metrics["macro_f1_3class"] = round(
            f1_score(gt_3class, pred_3class, average="macro",
                     labels=LABELS_3CLASS, zero_division=0), 4
        )
        metrics["weighted_f1_3class"] = round(
            f1_score(gt_3class, pred_3class, average="weighted",
                     labels=LABELS_3CLASS, zero_division=0), 4
        )
        metrics["per_class_f1_3class"] = {
            label: round(
                f1_score(
                    [1 if g == label else 0 for g in gt_3class],
                    [1 if p == label else 0 for p in pred_3class],
                    zero_division=0,
                ), 4
            )
            for label in LABELS_3CLASS
        }
        metrics["classification_report_3class"] = classification_report(
            gt_3class, pred_3class, labels=LABELS_3CLASS, zero_division=0
        )
        metrics["confusion_matrix_3class"] = confusion_matrix(
            gt_3class, pred_3class, labels=LABELS_3CLASS
        ).tolist()

    pred_4class_oracle = []
    for r in results:
        p3 = r["predicted_3class"]
        gt = r["ground_truth_label"]
        if p3 == "Uncertain":
            pred_4class_oracle.append(gt if gt in [
                "Conflicting Evidence/Cherrypicking", "Not Enough Evidence"
            ] else "Not Enough Evidence")
        elif p3 == "Supported":
            pred_4class_oracle.append("Supported")
        else:
            pred_4class_oracle.append("Refuted")

    if HAS_SKLEARN:
        metrics["macro_f1_4class_oracle"] = round(
            f1_score(gt_4class, pred_4class_oracle, average="macro",
                     labels=AVERITEC_LABELS, zero_division=0), 4
        )
        metrics["accuracy_4class_oracle"] = round(
            accuracy_score(gt_4class, pred_4class_oracle), 4
        )

    correct = sum(1 for r in results if r["predicted_3class"] == AVERITEC_TO_3CLASS[r["ground_truth_label"]])
    metrics["simplified_averitec_score"] = round(correct / len(results), 4) if results else 0.0
    metrics["note_averitec_score"] = (
        "Simplified: counts label-correct predictions only. "
        "The paper's Averitec Score additionally requires Q+A METEOR >= 0.25, "
        "which SIFACT does not produce."
    )

    from collections import Counter
    metrics["ground_truth_distribution"] = dict(Counter(gt_4class))
    metrics["prediction_distribution"]   = dict(Counter(pred_3class))

    return metrics


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_report(metrics: dict, n_samples: int, mode: str) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  SIFACT Evaluation Report  |  n={n_samples}  |  mode={mode}")
    print(sep)

    print(f"\nPRIMARY METRICS (3-class: Supported / Refuted / Uncertain)")
    print(f"  Accuracy (3-class)        : {metrics.get('accuracy_3class', 'N/A')}")
    print(f"  Macro F1  (3-class)       : {metrics.get('macro_f1_3class', 'N/A')}")
    print(f"  Weighted F1 (3-class)     : {metrics.get('weighted_f1_3class', 'N/A')}")

    print(f"\nPER-CLASS F1 (3-class)")
    for label, score in metrics.get("per_class_f1_3class", {}).items():
        print(f"  {label:<38}: {score}")

    print(f"\nORACLE 4-CLASS (upper bound)")
    print(f"  Accuracy (4-class oracle) : {metrics.get('accuracy_4class_oracle', 'N/A')}")
    print(f"  Macro F1 (4-class oracle) : {metrics.get('macro_f1_4class_oracle', 'N/A')}")

    print(f"\nSIMPLIFIED AVERITEC SCORE")
    print(f"  Score                     : {metrics.get('simplified_averitec_score', 'N/A')}")
    print(f"  Note: {metrics.get('note_averitec_score', '')}")

    print(f"\nCONFUSION MATRIX (3-class)  rows=GT  cols=Pred")
    cm = metrics.get("confusion_matrix_3class")
    labels = ["Supported", "Refuted", "Uncertain"]
    if cm:
        header = "              " + "  ".join(f"{l[:8]:>10}" for l in labels)
        print(header)
        for i, row in enumerate(cm):
            print(f"  {labels[i]:<12}  " + "  ".join(f"{v:>10}" for v in row))

    print(f"\nCLASSIFICATION REPORT (3-class)")
    print(metrics.get("classification_report_3class", "N/A"))

    print(f"\nLABEL DISTRIBUTIONS")
    print("  Ground Truth:", metrics.get("ground_truth_distribution", {}))
    print("  Predictions: ", metrics.get("prediction_distribution", {}))

    print(f"\nCOMPARISON WITH FEVER PAPER BASELINE")
    print("  Paper baseline (Bloom + BERT, test set):")
    print("    Averitec Score : 0.11")
    print("  Paper best (Mixtral 8x22B, test set):")
    print("    Averitec Score : 0.33")
    print(f"  Your SIFACT simplified score: {metrics.get('simplified_averitec_score', 'N/A')}")
    print(f"\n{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global key_manager

    parser = argparse.ArgumentParser(
        description="Evaluate SIFACT against the AVeriTeC dataset"
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to AVeriTeC JSON file")
    parser.add_argument("--output", default="eval_results",
                        help="Directory to write results and checkpoint")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N samples")
    parser.add_argument("--direct-verify", action="store_true",
                        help="Bypass extraction agent — feed claims directly to verifier")
    parser.add_argument("--delay", type=float, default=API_DELAY_SECONDS,
                        help=f"Seconds between API calls (default: {API_DELAY_SECONDS})")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to existing checkpoint file to resume from")
    args = parser.parse_args()

    output_dir      = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or str(output_dir / "checkpoint.json")
    results_path    = str(output_dir / "results.json")
    metrics_path    = str(output_dir / "metrics.json")
    report_path     = str(output_dir / "report.txt")
    mode            = "direct_verify" if args.direct_verify else "full_pipeline"

    logger.info("Evaluation mode: %s", mode)

    # ── Initialise API key manager ────────────────────────────────────────────
    key_manager = APIKeyManager(initial_key=GROQ_API_KEY, model=VERIFICATION_MODEL)
    logger.info("API key manager initialised.")

    # ── Load dataset ──────────────────────────────────────────────────────────
    logger.info("Loading dataset from %s …", args.dataset)
    records = load_dataset(args.dataset)
    if args.limit:
        records = records[: args.limit]
    logger.info("Loaded %d records", len(records))

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpoint    = load_checkpoint(checkpoint_path)
    results: list[dict] = checkpoint["results"]
    processed_ids: set  = set(checkpoint["processed_ids"])
    logger.info("Resuming: %d already processed", len(processed_ids))

    # ── Evaluation loop ───────────────────────────────────────────────────────
    iterator = enumerate(records)
    if HAS_TQDM:
        iterator = tqdm(iterator, total=len(records), desc="Evaluating")

    user_stopped = False  # track whether the user chose to abort after a rate-limit

    for idx, record in iterator:
        record_id    = record.get("id", str(idx))
        if record_id in processed_ids:
            continue

        claim_text   = record.get("claim", "").strip()
        ground_truth = extract_label(record)

        if not claim_text:
            logger.warning("Skipping record %s — empty claim", record_id)
            continue

        logger.info("[%d/%d] Claim: %s…", idx + 1, len(records), claim_text[:80])

        try:
            if args.direct_verify:
                output = run_direct_verify(claim_text, claim_id=record_id)
            else:
                # Full pipeline: wrap the graph invoke with rate-limit retry
                output = key_manager.run_with_retry(run_full_pipeline, claim_text)

            verdict     = output.get("final_verdict", "UNCERTAIN")
            confidence  = output.get("confidence_score", 0.0)
            explanation = output.get("explanation", "")

        except RuntimeError as exc:
            # User chose not to provide a new key — save and exit gracefully
            if "stopped by user" in str(exc).lower():
                logger.info("User aborted after rate limit. Saving checkpoint…")
                user_stopped = True
                break
            logger.error("Runtime error on record %s: %s", record_id, exc)
            verdict, confidence, explanation = "UNCERTAIN", 0.0, f"Error: {exc}"

        except Exception as exc:
            logger.error("Error on record %s: %s", record_id, exc)
            verdict, confidence, explanation = "UNCERTAIN", 0.0, f"Error: {exc}"

        predicted_3class = SIFACT_TO_3CLASS.get(verdict, "Uncertain")
        gt_3class        = AVERITEC_TO_3CLASS.get(ground_truth, "Uncertain")
        is_correct       = predicted_3class == gt_3class

        result_entry = {
            "id":                  record_id,
            "claim":               claim_text,
            "ground_truth_label":  ground_truth,
            "ground_truth_3class": gt_3class,
            "sifact_verdict":      verdict,
            "predicted_3class":    predicted_3class,
            "confidence":          confidence,
            "explanation":         explanation,
            "is_correct":          is_correct,
        }
        results.append(result_entry)
        processed_ids.add(record_id)

        logger.info(
            "  → %s (mapped: %s) | GT: %s | %s",
            verdict, predicted_3class, ground_truth,
            "✓ CORRECT" if is_correct else "✗ WRONG",
        )

        # Checkpoint every 10 samples
        if (idx + 1) % 10 == 0:
            save_checkpoint(
                {"results": results, "processed_ids": list(processed_ids)},
                checkpoint_path,
            )

        time.sleep(args.delay)

    # ── Final checkpoint ──────────────────────────────────────────────────────
    save_checkpoint({"results": results, "processed_ids": list(processed_ids)}, checkpoint_path)
    logger.info("Checkpoint saved to %s", checkpoint_path)

    if user_stopped:
        print("\n[INFO] Evaluation paused mid-run. Re-run the script with:")
        print(f"       --checkpoint {checkpoint_path}")
        print("       to resume from where you left off.\n")

    # ── Save raw results ──────────────────────────────────────────────────────
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Raw results saved to %s", results_path)

    if not results:
        print("[ERROR] No results collected — cannot compute metrics.")
        return

    # ── Compute and save metrics ──────────────────────────────────────────────
    metrics = compute_metrics(results)

    def _serialise(obj):
        if isinstance(obj, list) and obj and isinstance(obj[0], list):
            return [[int(v) for v in row] for row in obj]
        return obj

    metrics_serialisable = {
        k: (_serialise(v) if isinstance(v, list) else v)
        for k, v in metrics.items()
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_serialisable, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # ── Print and save report ─────────────────────────────────────────────────
    print_report(metrics, len(results), mode)

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(metrics, len(results), mode)
    with open(report_path, "w") as f:
        f.write(buf.getvalue())
    logger.info("Text report saved to %s", report_path)


if __name__ == "__main__":
    main()