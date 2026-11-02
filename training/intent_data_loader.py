import json
from pathlib import Path


# -----------------------------------------
# 1. PROJECT PATHS
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "intent"


# -----------------------------------------
# 2. DATASET FILES
# -----------------------------------------

FILES = {
    "train": DATA_DIR / "sentra_train_clean.jsonl",
    "test1": DATA_DIR / "sentra_test1_clean.jsonl",
    "test2": DATA_DIR / "sentra_test2_clean.jsonl"
}


# -----------------------------------------
# 3. REQUIRED DATASET FIELDS
# -----------------------------------------

REQUIRED_FIELDS = {
    "example_id",
    "group_id",
    "scenario_family",
    "conversation_context",
    "prompt",
    "intent_labels",
    "action",
    "target",
    "data_type",
    "sensitivity",
    "destination",
    "authorization",
    "context_dependency",
    "user_role"
}


# -----------------------------------------
# 4. LOAD JSONL DATASET
# -----------------------------------------

def load_jsonl(file_path):

    records = []

    with file_path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {file_path.name}, "
                    f"line {line_number}: {error}"
                )

            missing_fields = REQUIRED_FIELDS - record.keys()

            if missing_fields:
                raise ValueError(
                    f"{file_path.name}, line {line_number}: "
                    f"Missing fields: {sorted(missing_fields)}"
                )

            records.append(record)

    return records


# -----------------------------------------
# 5. LOAD ALL DATASET SPLITS
# -----------------------------------------

def load_all_datasets():

    datasets = {}

    for split_name, file_path in FILES.items():

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {file_path}"
            )

        datasets[split_name] = load_jsonl(file_path)

        print(
            f"{split_name}: "
            f"{len(datasets[split_name])} records loaded"
        )

    return datasets


# -----------------------------------------
# 6. DISPLAY SAMPLE RECORD
# -----------------------------------------

def display_sample(datasets):

    sample = datasets["train"][0]

    print("\nSample training record:")

    print(json.dumps(
        sample,
        indent=4,
        ensure_ascii=False
    ))


# -----------------------------------------
# 7. MAIN EXECUTION
# -----------------------------------------

if __name__ == "__main__":

    datasets = load_all_datasets()

    display_sample(datasets)

    print("\nDataset loading completed successfully!")