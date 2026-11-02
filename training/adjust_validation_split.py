
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = ROOT / "data" / "intent" / "split" / "train.jsonl"
VALIDATION_FILE = ROOT / "data" / "intent" / "split" / "validation.jsonl"

GROUPS_TO_MOVE = {"MIX_00227", "MIX_00407"}


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    train = load_jsonl(TRAIN_FILE)
    validation = load_jsonl(VALIDATION_FILE)

    moving = [
        record for record in validation
        if record["group_id"] in GROUPS_TO_MOVE
    ]

    found_groups = {record["group_id"] for record in moving}

    if found_groups != GROUPS_TO_MOVE:
        raise ValueError(
            f"Expected groups {GROUPS_TO_MOVE}, found {found_groups}"
        )

    if any(record["group_id"] in GROUPS_TO_MOVE for record in train):
        raise ValueError("A selected group already exists in training.")

    remaining_validation = [
        record for record in validation
        if record["group_id"] not in GROUPS_TO_MOVE
    ]

    updated_train = train + moving

    save_jsonl(TRAIN_FILE, updated_train)
    save_jsonl(VALIDATION_FILE, remaining_validation)

    print("GROUP SPLIT UPDATED")
    print("Moved groups:", sorted(found_groups))
    print("Moved records:", len(moving))
    print("Training records:", len(updated_train))
    print("Validation records:", len(remaining_validation))


if __name__ == "__main__":
    main()
