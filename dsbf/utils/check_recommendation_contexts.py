# dsbf/utils/check_recommendation_contexts.py

import ast
import re
from pathlib import Path

import yaml

RECO_PATH = Path("dsbf/config/recommendation_library.yaml")
TASKS_PATH = Path("dsbf/eda/tasks")


def extract_variables_from_condition(expr: str):
    """Extract all variable names used in the YAML condition string."""
    try:
        tree = ast.parse(expr, mode="eval")
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    except Exception:
        return set()


def load_yaml_rules():
    with open(RECO_PATH) as f:
        tips = yaml.safe_load(f)

    lookup = {}  # task_name -> required_vars
    for tip_id, tip in tips.items():
        for task in tip.get("applicable_to", []):
            vars_needed = extract_variables_from_condition(tip["condition"])
            lookup.setdefault(task, set()).update(vars_needed)
    return lookup


def extract_vars_passed_to_tip(filepath):
    with open(filepath) as f:
        src = f.read()

    matches = re.findall(r"get_recommendation_tip\([^)]+?,\s*{(.*?)}\)", src, re.DOTALL)
    used_vars = set()
    for block in matches:
        var_matches = re.findall(r"['\"](\w+)['\"]\s*:", block)
        used_vars.update(var_matches)
    return used_vars


def main():
    task_context_requirements = load_yaml_rules()

    print("🔎 Recommendation context audit:\n")
    for task_name, expected_vars in sorted(task_context_requirements.items()):
        file_path = TASKS_PATH / f"{task_name}.py"
        if not file_path.exists():
            print(f"⚠️  Task file not found: {task_name}")
            continue

        actual_vars = extract_vars_passed_to_tip(file_path)
        missing = expected_vars - actual_vars
        if missing:
            print(f"❌ {task_name}: missing {sorted(missing)}")
        else:
            print(f"✅ {task_name}: all required vars present")

    print("\n✅ Audit complete.")


if __name__ == "__main__":
    main()
