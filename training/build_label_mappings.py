
import json
from pathlib import Path

DATASET = Path("data/intent/sentra_train_clean.jsonl")
OUTPUT = Path("configs/label_mappings.json")

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

# Collect labels using only the training dataset.
labels = {field: set() for field in FIELDS}

with DATASET.open("r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)

        for field in FIELDS:
            value = record[field]

            if isinstance(value, list):
                labels[field].update(value)
            else:
                labels[field].add(value)

# Create deterministic alphabetical mappings.
mappings = {}

for field in FIELDS:
    if field == "context_dependency":
        mappings[field] = {
            "False": 0,
            "True": 1
        }
    else:
        sorted_labels = sorted(labels[field])
        mappings[field] = {
            label: index
            for index, label in enumerate(sorted_labels)
        }

# Save mappings.
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as file:
    json.dump(mappings, file, indent=4)

print("\n===== SENTRA LABEL MAPPINGS =====")

for field, mapping in mappings.items():
    print(f"\n{field.upper()} ({len(mapping)} labels)")

    for label, index in mapping.items():
        print(f"{label:40} -> {index}")

print(f"\nMappings saved to: {OUTPUT}")
