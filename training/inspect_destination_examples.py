
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = ROOT / "data" / "intent" / "split" / "train.jsonl"

TARGET_DESTINATIONS = [
    "PUBLIC_WEBSITE",
    "EXTERNAL_AI_SERVICE",
    "PERSONAL_ACCOUNT",
    "EXTERNAL_PERSON"
]

examples = defaultdict(list)
group_counts = defaultdict(set)

with TRAIN_FILE.open("r", encoding="utf-8") as file:

    for line in file:

        record = json.loads(line)

        destination = record["destination"]

        if destination in TARGET_DESTINATIONS:

            examples[destination].append(record)
            group_counts[destination].add(record["group_id"])


print("\n===== DESTINATION TRAINING DATA =====")

for destination in TARGET_DESTINATIONS:

    records = examples[destination]

    print(f"\n{'=' * 65}")
    print("DESTINATION:", destination)
    print("Total examples:", len(records))
    print("Unique groups:", len(group_counts[destination]))

    print("\n--- PROMPT EXAMPLES ---")

    for index, record in enumerate(records[:15], start=1):

        print(f"\n{index}. Example ID: {record['example_id']}")
        print("Group ID:", record["group_id"])
        print("Prompt:", record["prompt"])
        print("Target:", record["target"])
        print("Data type:", record["data_type"])
        print("Intent:", record["intent_labels"])
        print("Action:", record["action"])
        print("Authorization:", record["authorization"])

    if len(records) > 15:
        print(f"\n... {len(records) - 15} more examples not displayed.")

print("\n===== DESTINATION COUNTS =====")

counts = Counter()

for destination in TARGET_DESTINATIONS:
    counts[destination] = len(examples[destination])

for destination, count in counts.most_common():
    print(f"{destination:25} {count}")
