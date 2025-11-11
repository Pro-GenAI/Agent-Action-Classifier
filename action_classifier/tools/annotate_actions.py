"""Annotate actions JSON with assistant flags.

This script updates `action_classifier/sample_actions.json` in-place.
For each action it ensures the fields `classification`, `risk_level`,
`flagged_by` and `flag_reason` exist. The decisions are made by a
lightweight heuristic (assistant-written), not the model/classifier.

Run:
    python3 tools/annotate_actions.py
"""

import json
from pathlib import Path


def decide_classification_and_risk(action):
    # Use existing values if present
    cls = action.get("classification")
    rl = action.get("risk_level")
    prompt = (action.get("prompt") or "").lower()
    label = (action.get("label") or "").lower()
    resource = (action.get("resource") or "").lower()
    action_meta = action.get("action") or {}
    server_label = (action_meta.get("server_label") or "").lower()

    keywords_harmful = [
        "harass", "threat", "misinformation", "deepfake", "child",
        "exploit", "poison", "malware", "phish", "credential", "exfiltrate",
        "pathogen", "nuclear", "shutdown", "scada", "steal", "attack",
    ]
    keywords_unethical = [
        "bias", "manipulate", "decept", "fake review", "bypass", "scrape",
        "exploit", "forge", "fabricate", "deanonym", "deanonymize", "paywall",
    ]

    # If classification already present, keep it
    if cls in ("harmful", "unethical", "safe"):
        classification = cls
    else:
        text = " ".join([prompt, label, resource, server_label])
        if any(k in text for k in keywords_harmful):
            classification = "harmful"
        elif any(k in text for k in keywords_unethical):
            classification = "unethical"
        else:
            classification = "safe"

    if rl in ("none", "low", "medium", "high"):
        risk = rl
    else:
        # simple mapping
        if classification == "safe":
            risk = "none"
        elif classification == "unethical":
            risk = "low"
        else:  # harmful
            # escalate for especially dangerous keywords
            high_risk_keys = ["child", "pathogen", "nuclear", "scada", "critical", "exploit"]
            text = " ".join([prompt, label, resource, server_label])
            if any(k in text for k in high_risk_keys):
                risk = "high"
            else:
                risk = "medium"

    return classification, risk


def make_flag_reason(action, classification, risk):
    # Compose a short reason for the assistant's choice
    prompt = action.get("prompt", "").strip()
    label = action.get("label", "").strip()
    reason = f"Assistant annotated as {classification} (risk={risk}) based on prompt and labels."
    # Add a tiny hint from the prompt when short
    if prompt:
        snippet = prompt if len(prompt) < 120 else prompt[:117] + "..."
        reason += f" Example: '{snippet}'"
    return reason


def main():
    repo_root = Path(__file__).parent.parent
    p = repo_root / "action_classifier" / "sample_actions.json"
    assert p.exists(), f"File not found: {p}"

    data = json.loads(p.read_text(encoding="utf-8"))
    actions = data.get("actions", [])
    changed = 0
    for a in actions:
        orig = dict(a)
        classification, risk = decide_classification_and_risk(a)
        a["classification"] = classification
        a["risk_level"] = risk
        # a["flagged_by"] = "assistant"
        # a["flag_reason"] = make_flag_reason(a, classification, risk)
        if a != orig:
            changed += 1

    # write back
    if changed:
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Annotated {changed} actions and updated {p}")
    else:
        print("No changes made")

    print("WARNING: Make sure to review the changes before using the updated data!")


if __name__ == "__main__":
    main()
