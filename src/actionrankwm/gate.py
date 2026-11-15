from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _check(
    name: str,
    actual: float | int | None,
    threshold: float | int,
    comparison: str = ">=",
) -> dict[str, Any]:
    if actual is None:
        return {
            "name": name,
            "status": "missing",
            "actual": None,
            "threshold": threshold,
            "comparison": comparison,
            "passed": False,
        }
    if comparison == ">=":
        passed = actual >= threshold
    elif comparison == "==":
        passed = actual == threshold
    else:
        raise ValueError(f"Unsupported gate comparison: {comparison}")
    return {
        "name": name,
        "status": "evaluated",
        "actual": actual,
        "threshold": threshold,
        "comparison": comparison,
        "passed": passed,
    }


def evaluate_gate(metrics: Mapping[str, Any], gate_config: Mapping[str, Any]) -> dict[str, Any]:
    decision = metrics["decision"]
    overall = decision["overall"]
    by_domain = decision["by_domain"]
    risk = metrics["risk"]
    uncertainty = metrics.get("uncertainty", {})
    scientific = bool(gate_config.get("scientific", True))

    harness_checks = [
        _check("domain_count", len(by_domain), int(gate_config["min_domains"])),
    ]
    for domain, values in sorted(by_domain.items()):
        harness_checks.append(
            _check(
                f"tasks_in_{domain}",
                int(values["task_count"]),
                int(gate_config["min_tasks_per_domain"]),
            )
        )

    if "min_high_fidelity_unsafe_top_sets" in gate_config:
        scientific_checks = [
            _check(
                "high_fidelity_unsafe_top_sets_overall",
                int(overall["high_text_fidelity_unsafe_top_set_count"]),
                int(gate_config["min_high_fidelity_unsafe_top_sets"]),
            )
        ]
        for domain, values in sorted(by_domain.items()):
            scientific_checks.append(
                _check(
                    f"high_fidelity_unsafe_top_sets_in_{domain}",
                    int(values["high_text_fidelity_unsafe_top_set_count"]),
                    int(gate_config["min_high_fidelity_unsafe_top_sets_per_domain"]),
                )
            )
    else:
        scientific_checks = [
            _check(
                "high_fidelity_harmful_flips_overall",
                int(overall["high_text_fidelity_harmful_flip_count"]),
                int(gate_config["min_high_fidelity_harmful_flips"]),
            )
        ]
        for domain, values in sorted(by_domain.items()):
            scientific_checks.append(
                _check(
                    f"high_fidelity_harmful_flips_in_{domain}",
                    int(values["high_text_fidelity_harmful_flip_count"]),
                    int(gate_config["min_high_fidelity_harmful_flips_per_domain"]),
                )
            )
    scientific_checks.extend(
        [
            _check(
                "decision_correlation_advantage",
                overall["decision_correlation_advantage"],
                float(gate_config["min_decision_correlation_advantage"]),
            ),
            _check(
                "risk_auroc",
                risk.get("proposed_auroc"),
                float(gate_config["min_risk_auroc"]),
            ),
            _check(
                "risk_auroc_advantage",
                risk.get("proposed_auroc_advantage"),
                float(gate_config["min_risk_auroc_advantage"]),
            ),
        ]
    )
    if "min_decision_correlation_advantage_ci_low" in gate_config:
        advantage_interval = (
            uncertainty.get("overall", {})
            .get("intervals", {})
            .get("decision_correlation_advantage", {})
        )
        scientific_checks.append(
            _check(
                "decision_correlation_advantage_ci_low",
                advantage_interval.get("ci_low"),
                float(gate_config["min_decision_correlation_advantage_ci_low"]),
            )
        )
    routing = risk.get("routing", {})
    harmful_reduction = routing.get("harmful_reduction")
    success_gain = routing.get("terminal_success_gain")
    routing_passed = (
        harmful_reduction is not None
        and harmful_reduction >= float(gate_config["min_harmful_reduction"])
    ) or (
        success_gain is not None
        and success_gain >= float(gate_config["min_success_gain"])
    )
    scientific_checks.append(
        {
            "name": "routing_utility",
            "status": "evaluated"
            if harmful_reduction is not None or success_gain is not None
            else "missing",
            "actual": {
                "harmful_reduction": harmful_reduction,
                "terminal_success_gain": success_gain,
            },
            "threshold": {
                "harmful_reduction": float(gate_config["min_harmful_reduction"]),
                "terminal_success_gain": float(gate_config["min_success_gain"]),
                "logic": "OR",
            },
            "comparison": ">= either",
            "passed": routing_passed,
        }
    )

    harness_passed = all(item["passed"] for item in harness_checks)
    scientific_passed = all(item["passed"] for item in scientific_checks)
    if not harness_passed:
        status = "HARNESS_INVALID"
    elif not scientific:
        status = "CONTROL_COMPLETE"
    elif scientific_passed:
        status = "GO"
    elif any(item["status"] == "missing" for item in scientific_checks):
        status = "NO_GO_INCOMPLETE"
    else:
        status = "NO_GO"
    return {
        "status": status,
        "scientific_gate": scientific,
        "harness_passed": harness_passed,
        "scientific_passed": scientific_passed if scientific else None,
        "harness_checks": harness_checks,
        "scientific_checks": scientific_checks,
        "interpretation": (
            "Control run validates mechanics only; it is not scientific evidence."
            if not scientific and harness_passed
            else "All frozen scientific checks passed."
            if status == "GO"
            else "At least one frozen check failed or could not be evaluated."
        ),
    }


def gate_markdown(decision: Mapping[str, Any]) -> str:
    lines = [
        "# Gate decision",
        "",
        f"**Status:** `{decision['status']}`",
        "",
        str(decision["interpretation"]),
        "",
        "## Harness checks",
        "",
        "| Check | Actual | Threshold | Pass |",
        "|---|---:|---:|:---:|",
    ]
    for item in decision["harness_checks"]:
        lines.append(
            f"| {item['name']} | {item['actual']} | {item['comparison']} {item['threshold']} | {'yes' if item['passed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Scientific checks",
            "",
            "These checks are reported for controls but required only when `scientific=true`.",
            "",
            "| Check | Actual | Threshold | Pass |",
            "|---|---|---|:---:|",
        ]
    )
    for item in decision["scientific_checks"]:
        lines.append(
            f"| {item['name']} | {item['actual']} | {item['comparison']} {item['threshold']} | {'yes' if item['passed'] else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)
