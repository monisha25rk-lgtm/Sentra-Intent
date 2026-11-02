
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]

ERROR_FILE = ROOT / "logs" / "test1_error_analysis.jsonl"
TEST_FILE = ROOT / "data" / "intent" / "encoded" / "test1_encoded.jsonl"

# Find groups containing scenario_family errors.
affected_groups = set()

with open(ERROR_FILE, "r", encoding="utf-8") as file:
    for line in file:
        error = json.loads(line)

        if "scenario_family" in error["incorrect_fields"]:
            affected_groups.add(error["group_id"])

groups = defaultdict(list)

with open(TEST_FILE, "r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)
        original = record["original_record"]

        if original["group_id"] in affected_groups:
            groups[original["group_id"]].append(original)

print("=" * 75)
print("COMPLETE AFFECTED GROUP INSPECTION")
print("=" * 75)

print("Groups containing scenario-family errors:", len(affected_groups))

for group_id in sorted(affected_groups):
    print("\n" + "=" * 75)
    print("GROUP:", group_id)
    print("Total examples:", len(groups[group_id]))
    print("=" * 75)

    for record in groups[group_id]:
        print("\nExample:", record["example_id"])
        print("Scenario family:", record["scenario_family"])
        print("Intent:", record["intent_labels"])
        print("Action:", record["action"])
        print("Target:", record["target"])
        print("Data type:", record["data_type"])
        print("Destination:", record["destination"])
        print("Conversation context:", record["conversation_context"])

print("\nINSPECTION COMPLETED")
