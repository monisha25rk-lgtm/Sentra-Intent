import json
from collections import Counter

DATASET = "data/intent/sentra_train_clean.jsonl"

FIELDS = [
    "scenario_family",
    "intent_labels",
    "action",
    "target",
    "data_type",
    "destination",
    "authorization",
    "context_dependency"
]

label_counts = {field: Counter() for field in FIELDS}

with open(DATASET, "r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)

        for field in FIELDS:
            value = record[field]

            if isinstance(value, list):
                label_counts[field].update(value)
            else:
                label_counts[field].update([value])

print("\n===== SENTRA TRAINING LABEL INSPECTION =====")

for field in FIELDS:
    print(f"\n--- {field.upper()} ---")

    for label, count in label_counts[field].most_common():
        print(f"{str(label):35} {count}")