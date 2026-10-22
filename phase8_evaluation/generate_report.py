
"""
Phase 8 — Generate Answer Quality Report

Reads:
    reports/phase8/answer_quality_report.json
    phase8_evaluation/answer_quality_dataset.json

Creates:
    reports/phase8/answer_quality_summary.md

Features:
    - Overall pass/fail statistics
    - Average scores for all nine metrics
    - Category-level performance
    - Most frequently failed thresholds
    - Failed questions and errors

This script does not call NVIDIA or modify Qdrant.
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = ROOT / "reports" / "phase8"

INPUT = REPORT_DIR / "answer_quality_report.json"

DATASET = (
    ROOT
    / "phase8_evaluation"
    / "answer_quality_dataset.json"
)

OUTPUT = REPORT_DIR / "answer_quality_summary.md"


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def load_json(path):
    """Load a JSON file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def percent(numerator, denominator):
    """Calculate percentage safely."""

    if denominator == 0:
        return 0.0

    return (numerator / denominator) * 100


# --------------------------------------------------
# Generate report
# --------------------------------------------------

def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Run evaluate_answer_quality first: {INPUT}"
        )

    report = load_json(INPUT)

    results = report.get("results", [])

    if not isinstance(results, list):
        raise ValueError(
            "Invalid report: results must be a list."
        )

    # Load dataset metadata
    dataset = (
        load_json(DATASET)
        if DATASET.exists()
        else []
    )

    metadata = {
        case["id"]: case
        for case in dataset
        if isinstance(case, dict)
        and "id" in case
    }

    # --------------------------------------------------
    # Separate completed evaluations and errors
    # --------------------------------------------------

    completed = [
        result
        for result in results
        if isinstance(result, dict)
        and isinstance(
            result.get("evaluation"), dict
        )
    ]

    errors = [
        result
        for result in results
        if isinstance(result, dict)
        and "evaluation" not in result
    ]

    passed = [
        result
        for result in completed
        if result["evaluation"].get("pass") is True
    ]

    failed = [
        result
        for result in completed
        if result["evaluation"].get("pass") is not True
    ]

    # --------------------------------------------------
    # Group results by category
    # --------------------------------------------------

    category_groups = defaultdict(list)

    for result in completed:

        case_id = result.get("id")

        category = metadata.get(
            case_id, {}
        ).get(
            "category",
            result.get("topic", "unknown")
        )

        category_groups[category].append(result)

    # --------------------------------------------------
    # Calculate metric averages and failure counts
    # --------------------------------------------------

    metric_values = defaultdict(list)

    threshold_failures = Counter()

    for result in completed:

        scores = result.get("scores", {})

        for metric, score in scores.items():

            if (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
            ):
                metric_values[metric].append(
                    float(score)
                )

        failed_metrics = (
            result["evaluation"].get(
                "failed_metrics", {}
            )
        )

        for metric in failed_metrics:
            threshold_failures[metric] += 1

    # --------------------------------------------------
    # Report overview
    # --------------------------------------------------

    lines = [
        "# Ireland Navigator AI — Phase 8 Answer Quality Report",
        "",
        (
            "Generated (UTC): "
            + datetime.now(
                timezone.utc
            ).isoformat(timespec="seconds")
        ),
        "",
        "## Overview",
        "",
        f"- Dataset questions: {len(dataset)}",
        f"- Report entries: {len(results)}",
        f"- Evaluated: {len(completed)}",
        f"- Passed: {len(passed)}",
        f"- Failed: {len(failed)}",
        f"- Errors / unevaluated: {len(errors)}",
        (
            "- Pass rate (evaluated only): "
            f"{percent(len(passed), len(completed)):.1f}%"
        ),
        "",
        "## Mean metric scores (1–5)",
        "",
        "| Metric | Mean | Number scored |",
        "|---|---:|---:|",
    ]

    # --------------------------------------------------
    # Metric scores table
    # --------------------------------------------------

    for metric in sorted(metric_values):

        values = metric_values[metric]

        average = sum(values) / len(values)

        lines.append(
            f"| {metric} | {average:.2f} "
            f"| {len(values)} |"
        )

    # --------------------------------------------------
    # Results by category
    # --------------------------------------------------

    lines.extend([
        "",
        "## Results by category",
        "",
        "| Category | Evaluated | Passed | Pass rate |",
        "|---|---:|---:|---:|",
    ])

    for category, items in sorted(
        category_groups.items()
    ):

        category_passes = sum(
            item["evaluation"].get("pass") is True
            for item in items
        )

        pass_rate = percent(
            category_passes,
            len(items)
        )

        lines.append(
            f"| {category} | {len(items)} "
            f"| {category_passes} "
            f"| {pass_rate:.1f}% |"
        )

    # --------------------------------------------------
    # Most frequent threshold failures
    # --------------------------------------------------

    lines.extend([
        "",
        "## Most frequent threshold failures",
        "",
    ])

    if threshold_failures:

        for metric, count in (
            threshold_failures.most_common()
        ):

            lines.append(
                f"- {metric}: {count} case(s)"
            )

    else:

        lines.append(
            "No metric threshold failures "
            "in completed evaluations."
        )

    # --------------------------------------------------
    # Failed questions and errors
    # --------------------------------------------------

    lines.extend([
        "",
        "## Failed questions and errors",
        "",
    ])

    if not failed and not errors:
        lines.append(
            "No failures or errors recorded."
        )

    for result in failed:

        evaluation = result["evaluation"]

        failed_metric_names = (
            evaluation.get(
                "failed_metrics", {}
            ).keys()
        )

        reasons = (
            ", ".join(failed_metric_names)
            or "Other pass gate"
        )

        lines.append(
            f"- **{result.get('id', '?')}** — "
            f"{result.get('question', '?')} | "
            f"{evaluation.get('overall_score', '?')}% "
            f"| {reasons}"
        )

    for result in errors:

        lines.append(
            f"- **{result.get('id', '?')}** — "
            "ERROR: "
            f"{result.get('error', 'Unknown error')}"
        )

    # --------------------------------------------------
    # Interpretation limitations
    # --------------------------------------------------

    lines.extend([
        "",
        "## Interpretation limits",
        "",
        (
            "- Scores are model-judge assessments, "
            "not independently verified accuracy rates."
        ),
        (
            "- Human review is recommended because "
            "the same model family generates "
            "and evaluates answers."
        ),
        (
            "- Unsupported questions require "
            "refusal-aware citation rules."
        ),
        (
            "- Category-level findings require "
            "sufficient test cases and verified "
            "reference answers."
        ),
        "",
    ])

    # --------------------------------------------------
    # Save report
    # --------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("=" * 60)
    print("PHASE 8 — REPORT GENERATED")
    print("=" * 60)

    print(f"Report: {OUTPUT}")

    print(f"Evaluated: {len(completed)}")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    print(f"Errors: {len(errors)}")

    print(
        "Pass rate: "
        f"{percent(len(passed), len(completed)):.1f}%"
    )


if __name__ == "__main__":
    main()
