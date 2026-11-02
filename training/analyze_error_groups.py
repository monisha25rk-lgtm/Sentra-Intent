
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]

ERROR_FILE = ROOT / "logs" / "test1_error_analysis.jsonl"

groups = defaultdict(list)
field_counts = Counter()
confusion_counts = Counter()

with open(ERROR_FILE, "r", encoding="utf-8") as file:
    errors = [json.loads(line) for line in file if line.strip()]

for error in errors:
    group_id = error["group_id"]
    groups[group_id].append(error)

    for field, values in error["incorrect_fields"].items():
        field_counts[field] += 1

        if field == "scenario_family":
            confusion_counts[
                (values["actual"], values["predicted"])
            ] += 1

print("=" * 70)
print("SENTRA GROUP-LEVEL ERROR ANALYSIS")
print("=" * 70)

print("\nTotal incorrect records:", len(errors))
print("Unique groups containing errors:", len(groups))

print("\n===== TOP GROUPS BY ERROR COUNT =====")

for group_id, records in sorted(
    groups.items(),
    key=lambda item: len(item[1]),
    reverse=True
)[:15]:
    print(f"\nGroup: {group_id}")
    print("Incorrect records:", len(records))

    for record in records:
        print("  Example:", record["example_id"])
        print("  Prompt:", record["prompt"])

        if "scenario_family" in record["incorrect_fields"]:
            values = record["incorrect_fields"]["scenario_family"]
            print("  Actual:", values["actual"])
            print("  Predicted:", values["predicted"])

print("\n===== SCENARIO CONFUSIONS =====")

for (actual, predicted), count in confusion_counts.most_common():
    print(f"{actual} -> {predicted}: {count}")

print("\n===== FIELD ERROR COUNTS =====")

for field, count in field_counts.most_common():
    print(f"{field:<25} {count}")

print("\nANALYSIS COMPLETED")
