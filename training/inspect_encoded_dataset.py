
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "intent" / "encoded"

SPLITS = [
    "train_encoded.jsonl",
    "test1_encoded.jsonl",
    "test2_encoded.jsonl"
]

MULTI_FIELDS = [
    "intent_labels",
    "action",
    "data_type"
]

SINGLE_FIELDS = [
    "scenario_family",
    "target",
    "destination",
    "authorization",
    "context_dependency"
]

for filename in SPLITS:

    path = DATA_DIR / filename

    lengths = []
    records = 0
    invalid_tokens = 0
    invalid_labels = 0

    with path.open("r", encoding="utf-8") as file:

        for line in file:
            record = json.loads(line)
            records += 1

            ids = record["input_ids"]
            lengths.append(len(ids))

            if any(token < 0 or token >= 1988 for token in ids):
                invalid_tokens += 1

            labels = record["labels"]

            for field in MULTI_FIELDS:
                if not isinstance(labels[field], list):
                    invalid_labels += 1

            for field in SINGLE_FIELDS:
                if not isinstance(labels[field], int):
                    invalid_labels += 1

    lengths.sort()

    print(f"\n===== {filename.upper()} =====")
    print("Records:", records)
    print("Minimum length:", min(lengths))
    print("Median length:", statistics.median(lengths))
    print("95th percentile:", lengths[int(0.95 * (len(lengths) - 1))])
    print("Maximum length:", max(lengths))
    print("Average length:", round(statistics.mean(lengths), 2))
    print("Invalid token IDs:", invalid_tokens)
    print("Invalid label structures:", invalid_labels)

    if filename == "train_encoded.jsonl":
        with path.open("r", encoding="utf-8") as file:
            sample = json.loads(next(file))

        print("\nSample record keys:", list(sample.keys()))
        print("Original record fields:", len(sample["original_record"]))
        print("Sample input IDs:", sample["input_ids"])
        print("Sample labels:", sample["labels"])
