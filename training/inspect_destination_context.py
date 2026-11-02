
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = ROOT / "data" / "intent" / "split" / "train.jsonl"

DESTINATIONS = [
    "PUBLIC_WEBSITE",
    "EXTERNAL_AI_SERVICE",
    "PERSONAL_ACCOUNT",
    "EXTERNAL_PERSON"
]

examples = defaultdict(list)
context_counts = defaultdict(lambda: {"with_context": 0, "without_context": 0})

with TRAIN_FILE.open("r", encoding="utf-8") as file:

    for line in file:

        record = json.loads(line)
        destination = record["destination"]

        if destination not in DESTINATIONS:
            continue

        context = record.get("conversation_context", [])

        if context:
            context_counts[destination]["with_context"] += 1
            examples[destination].append(record)
        else:
            context_counts[destination]["without_context"] += 1


for destination in DESTINATIONS:

    print("\n" + "=" * 75)
    print("DESTINATION:", destination)

    counts = context_counts[destination]

    print("Examples with context:", counts["with_context"])
    print("Examples without context:", counts["without_context"])

    print("\n--- CONTEXT EXAMPLES ---")

    for index, record in enumerate(examples[destination][:8], start=1):

        print(f"\nExample {index}")
        print("Example ID:", record["example_id"])
        print("Group ID:", record["group_id"])

        print("Conversation context:")
        print(json.dumps(
            record["conversation_context"],
            indent=2,
            ensure_ascii=False
        ))

        print("Current prompt:", record["prompt"])
        print("Expected destination:", record["destination"])

print("\nInspection completed.")
