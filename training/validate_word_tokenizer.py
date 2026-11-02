
import sys
import json
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "tokenizer")
)

from word_bpe import (
    load_tokenizer,
    encode,
    decode
)

DATA_DIR = PROJECT_ROOT / "data" / "intent"

MERGES_PATH = (
    PROJECT_ROOT
    / "tokenizer"
    / "merges_word_v3.txt"
)

DATA_FILES = [
    "sentra_train_clean.jsonl",
    "sentra_test1_clean.jsonl",
    "sentra_test2_clean.jsonl"
]


def main():

    merges, vocabulary, vocab_size = load_tokenizer(
        MERGES_PATH
    )

    print("\n===== SENTRA V3 VALIDATION =====")
    print("Learned merges:", len(merges))
    print("Vocabulary size:", vocab_size)

    total_records = 0
    total_tokens = 0
    total_characters = 0
    failed_roundtrips = 0
    invalid_ids = 0
    lengths = []

    for filename in DATA_FILES:

        file_path = DATA_DIR / filename

        record_count = 0
        token_count = 0
        character_count = 0
        split_lengths = []

        with file_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                if not line.strip():
                    continue

                record = json.loads(line)

                context = json.dumps(
                    record["conversation_context"],
                    ensure_ascii=False
                )

                model_input = (
                    "[CONTEXT] "
                    + context
                    + " [PROMPT] "
                    + record["prompt"]
                )

                token_ids = encode(
                    model_input,
                    merges
                )

                reconstructed = decode(
                    token_ids,
                    vocabulary
                )

                if reconstructed != model_input:
                    failed_roundtrips += 1

                for token_id in token_ids:
                    if token_id not in vocabulary:
                        invalid_ids += 1

                current_length = len(token_ids)

                record_count += 1
                token_count += current_length
                character_count += len(model_input)

                split_lengths.append(current_length)

        total_records += record_count
        total_tokens += token_count
        total_characters += character_count
        lengths.extend(split_lengths)

        print("\nDataset:", filename)
        print("Records:", record_count)
        print("Total tokens:", token_count)
        print(
            "Average tokens per record:",
            round(mean(split_lengths), 2)
        )
        print("Maximum tokens:", max(split_lengths))
        print("Minimum tokens:", min(split_lengths))

    print("\n===== OVERALL RESULTS =====")
    print("Total records:", total_records)
    print("Total tokens:", total_tokens)
    print("Total characters:", total_characters)
    print("Failed round trips:", failed_roundtrips)
    print("Invalid token IDs:", invalid_ids)
    print("Average tokens per record:", round(mean(lengths), 2))
    print("Maximum tokens:", max(lengths))

    print(
        "Average characters per token:",
        round(total_characters / total_tokens, 3)
    )

    sorted_lengths = sorted(lengths)
    p95 = sorted_lengths[
        int(0.95 * (len(sorted_lengths) - 1))
    ]

    print("95th percentile:", p95)

    if failed_roundtrips == 0 and invalid_ids == 0:
        print("\nPASS: V3 validation successful!")
    else:
        print("\nFAIL: Validation errors detected!")


if __name__ == "__main__":
    main()
