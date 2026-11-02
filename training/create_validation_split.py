
import json
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "intent" / "sentra_train_clean.jsonl"
)

OUTPUT_DIR = ROOT / "data" / "intent" / "split"


def main():

    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    groups = [record["group_id"] for record in records]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_indices, val_indices = next(
        splitter.split(records, groups=groups)
    )

    train_records = [records[i] for i in train_indices]
    val_records = [records[i] for i in val_indices]

    train_groups = {
        record["group_id"] for record in train_records
    }

    val_groups = {
        record["group_id"] for record in val_records
    }

    overlap = train_groups.intersection(val_groups)

    if overlap:
        raise ValueError(
            f"Group leakage detected: {overlap}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "train": train_records,
        "validation": val_records
    }

    for name, split_records in outputs.items():

        output_file = OUTPUT_DIR / f"{name}.jsonl"

        with output_file.open("w", encoding="utf-8") as file:

            for record in split_records:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        print(
            f"{name.upper()}: "
            f"{len(split_records)} records, "
            f"{len({r['group_id'] for r in split_records})} groups"
        )

    print("\nTotal records:", len(records))
    print("Group overlap:", len(overlap))
    print("Validation split created successfully.")


if __name__ == "__main__":
    main()
