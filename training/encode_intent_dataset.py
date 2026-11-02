
import json
import sys
from pathlib import Path

# Project directory
ROOT = Path(__file__).resolve().parents[1]

# Make tokenizer modules importable
sys.path.insert(0, str(ROOT / "tokenizer"))

from hybrid_bpe import (
    load_hybrid_tokenizer,
    encode_hybrid,
    decode_hybrid
)

# Paths
DATA_DIR = ROOT / "data" / "intent"
OUTPUT_DIR = DATA_DIR / "encoded"
CONFIG_DIR = ROOT / "configs"

WORD_MERGES = ROOT / "tokenizer" / "merges_word_v3.txt"
PHRASE_MERGES = ROOT / "tokenizer" / "merges_phrase_v4.txt"

MAPPING_FILE = CONFIG_DIR / "label_mappings.json"

SPLITS = {
    "train": DATA_DIR / "sentra_train_clean.jsonl",
    "test1": DATA_DIR / "sentra_test1_clean.jsonl",
    "test2": DATA_DIR / "sentra_test2_clean.jsonl"
}

MULTI_LABEL_FIELDS = [
    "intent_labels",
    "action",
    "data_type"
]

SINGLE_LABEL_FIELDS = [
    "scenario_family",
    "target",
    "destination",
    "authorization"
]


def load_label_mappings():
    with MAPPING_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def encode_record(
    record,
    word_merges,
    phrase_merges,
    vocabulary,
    mappings
):

    # Serialize context and prompt consistently.
    context_text = json.dumps(
        record["conversation_context"],
        ensure_ascii=False
    )

    model_input = (
        "[CONTEXT] "
        + context_text
        + " [PROMPT] "
        + record["prompt"]
    )

    # Tokenize using V4.
    input_ids = encode_hybrid(
        model_input,
        word_merges,
        phrase_merges
    )

    # Verify exact reconstruction.
    reconstructed = decode_hybrid(
        input_ids,
        vocabulary
    )

    if reconstructed != model_input:
        raise ValueError(
            f"Token reconstruction failed: "
            f"{record['example_id']}"
        )

    encoded_labels = {}

    # Multi-label fields: retain a list of IDs.
    for field in MULTI_LABEL_FIELDS:

        encoded_labels[field] = []

        for label in record[field]:

            if label not in mappings[field]:
                raise ValueError(
                    f"Unknown label '{label}' "
                    f"in field '{field}'"
                )

            encoded_labels[field].append(
                mappings[field][label]
            )

    # Single-label fields: one integer ID.
    for field in SINGLE_LABEL_FIELDS:

        label = record[field]

        if label not in mappings[field]:
            raise ValueError(
                f"Unknown label '{label}' "
                f"in field '{field}'"
            )

        encoded_labels[field] = mappings[field][label]

    # Boolean field.
    context_value = str(record["context_dependency"])

    if context_value not in mappings["context_dependency"]:
        raise ValueError(
            f"Unknown context_dependency: {context_value}"
        )

    encoded_labels["context_dependency"] = (
        mappings["context_dependency"][context_value]
    )

    # Preserve original record and all 14 fields.
    return {
        "example_id": record["example_id"],
        "group_id": record["group_id"],
        "input_ids": input_ids,
        "labels": encoded_labels,
        "original_record": record
    }


def process_split(
    split_name,
    dataset_path,
    word_merges,
    phrase_merges,
    vocabulary,
    mappings
):

    output_path = OUTPUT_DIR / f"{split_name}_encoded.jsonl"

    count = 0
    total_tokens = 0

    with dataset_path.open("r", encoding="utf-8") as source, \
         output_path.open("w", encoding="utf-8") as destination:

        for line in source:

            record = json.loads(line)

            encoded = encode_record(
                record,
                word_merges,
                phrase_merges,
                vocabulary,
                mappings
            )

            destination.write(
                json.dumps(encoded, ensure_ascii=False) + "\n"
            )

            count += 1
            total_tokens += len(encoded["input_ids"])

    print(f"\n{split_name.upper()}")
    print("Records:", count)
    print("Total tokens:", total_tokens)
    print("Average tokens per record:", round(total_tokens / count, 2))
    print("Saved:", output_path)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading V4 tokenizer...")

    (
        word_merges,
        phrase_merges,
        vocabulary,
        vocabulary_size
    ) = load_hybrid_tokenizer(
        WORD_MERGES,
        PHRASE_MERGES
    )

    print("Vocabulary size:", vocabulary_size)

    print("\nLoading label mappings...")

    mappings = load_label_mappings()

    print("Label mapping fields:", len(mappings))

    for split_name, dataset_path in SPLITS.items():

        process_split(
            split_name,
            dataset_path,
            word_merges,
            phrase_merges,
            vocabulary,
            mappings
        )

    print("\n===== DATASET ENCODING COMPLETED =====")


if __name__ == "__main__":
    main()
