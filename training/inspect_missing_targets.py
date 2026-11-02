
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = [
    ROOT / "data" / "intent" / "split" / "train.jsonl",
    ROOT / "data" / "intent" / "split" / "validation.jsonl"
]

TARGET_GROUPS = {"MIX_00227", "MIX_00407"}

for group_id in sorted(TARGET_GROUPS):
    print("\n" + "=" * 75)
    print("GROUP:", group_id)
    print("=" * 75)

    records = []

    for path in FILES:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)

                if record["group_id"] == group_id:
                    records.append(record)

    print("Total records in group:", len(records))

    for record in records:
        print("\nExample ID:", record["example_id"])
        print("Target:", record["target"])
        print("Intent:", record["intent_labels"])
        print("Prompt:", record["prompt"])

