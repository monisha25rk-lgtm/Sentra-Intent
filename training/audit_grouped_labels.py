
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = ROOT / "data" / "intent" / "encoded" / "train_grouped_encoded.jsonl"
VALIDATION_FILE = ROOT / "data" / "intent" / "encoded" / "validation_encoded.jsonl"
MAPPING_FILE = ROOT / "configs" / "label_mappings.json"

MULTI_LABEL_FIELDS = [
    "intent_labels",
    "action",
    "data_type"
]

SINGLE_LABEL_FIELDS = [
    "scenario_family",
    "target",
    "destination",
    "authorization",
    "context_dependency"
]


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    print("=" * 65)
    print("SENTRA GROUPED LABEL DISTRIBUTION AUDIT")
    print("=" * 65)

    train = load_jsonl(TRAIN_FILE)
    validation = load_jsonl(VALIDATION_FILE)

    with open(MAPPING_FILE, "r", encoding="utf-8") as file:
        mappings = json.load(file)

    print("\nTraining records:", len(train))
    print("Validation records:", len(validation))

    for field in MULTI_LABEL_FIELDS + SINGLE_LABEL_FIELDS:
        print("\n" + "=" * 65)
        print("FIELD:", field)
        print("=" * 65)

        mapping = mappings[field]
        inverse_mapping = {int(v): k for k, v in mapping.items()}

        train_counts = Counter()
        validation_counts = Counter()

        for record in train:
            labels = record["labels"][field]

            if field in MULTI_LABEL_FIELDS:
                train_counts.update(labels)
            else:
                train_counts.update([labels])

        for record in validation:
            labels = record["labels"][field]

            if field in MULTI_LABEL_FIELDS:
                validation_counts.update(labels)
            else:
                validation_counts.update([labels])

        missing = []
        rare = []

        for label_id, label_name in sorted(inverse_mapping.items()):
            train_count = train_counts[label_id]
            validation_count = validation_counts[label_id]

            print(
                f"{label_name:<45}"
                f"Train: {train_count:<6}"
                f"Validation: {validation_count}"
            )

            if train_count == 0:
                missing.append(label_name)
            elif train_count <= 5:
                rare.append((label_name, train_count))

        print("\nMISSING FROM TRAINING:", missing if missing else "NONE")
        print("RARE IN TRAINING (5 or fewer):", rare if rare else "NONE")

    print("\n" + "=" * 65)
    print("AUDIT COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
