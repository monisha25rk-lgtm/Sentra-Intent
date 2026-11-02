
import sys
import json
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "tokenizer")
)

from hybrid_bpe import (
    load_hybrid_tokenizer,
    encode_hybrid,
    decode_hybrid
)

DATA_DIR = PROJECT_ROOT / "data" / "intent"

WORD_MERGES_PATH = (
    PROJECT_ROOT / "tokenizer" / "merges_word_v3.txt"
)

PHRASE_MERGES_PATH = (
    PROJECT_ROOT / "tokenizer" / "merges_phrase_v4.txt"
)

DATA_FILES = [
    "sentra_train_clean.jsonl",
    "sentra_test1_clean.jsonl",
    "sentra_test2_clean.jsonl"
]


def main():

    (
        word_merges,
        phrase_merges,
        vocabulary,
        vocab_size
    ) = load_hybrid_tokenizer(
        WORD_MERGES_PATH,
        PHRASE_MERGES_PATH
    )

    print("\n===== SENTRA V4 VALIDATION =====")
    print("Word merges:", len(word_merges))
    print("Phrase merges:", len(phrase_merges))
    print("Vocabulary size:", vocab_size)

    total_records = 0
    total_tokens = 0
    total_characters = 0

    failed_roundtrips = 0
    invalid_ids = 0

    all_lengths = []

    for filename in DATA_FILES:

        file_path = DATA_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        records = 0
        tokens = 0
        characters = 0
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

                token_ids = encode_hybrid(
                    model_input,
                    word_merges,
                    phrase_merges
                )

                reconstructed = decode_hybrid(
                    token_ids,
                    vocabulary
                )

                if reconstructed != model_input:
                    failed_roundtrips += 1

                for token_id in token_ids:
                    if token_id not in vocabulary:
                        invalid_ids += 1

                current_length = len(token_ids)

                records += 1
                tokens += current_length
                characters += len(model_input)

                split_lengths.append(current_length)

        total_records += records
        total_tokens += tokens
        total_characters += characters

        all_lengths.extend(split_lengths)

        print("\n-----------------------------------")
        print("Dataset:", filename)
        print("Records:", records)
        print("Total tokens:", tokens)
        print(
            "Average tokens per record:",
            round(mean(split_lengths), 2)
        )
        print("Maximum tokens:", max(split_lengths))
        print("Minimum tokens:", min(split_lengths))

    sorted_lengths = sorted(all_lengths)

    p95_index = int(
        0.95 * (len(sorted_lengths) - 1)
    )

    print("\n========== OVERALL RESULTS ==========")

    print("Total records:", total_records)
    print("Total tokens:", total_tokens)
    print("Total characters:", total_characters)

    print("Failed round trips:", failed_roundtrips)
    print("Invalid token IDs:", invalid_ids)

    print(
        "Average tokens per record:",
        round(mean(all_lengths), 2)
    )

    print("Maximum tokens:", max(all_lengths))
    print("Minimum tokens:", min(all_lengths))
    print("95th percentile:", sorted_lengths[p95_index])

    print(
        "Average characters per token:",
        round(total_characters / total_tokens, 3)
        if total_tokens else 0
    )

    print("\n========== VALIDATION STATUS ==========")

    if failed_roundtrips == 0 and invalid_ids == 0:
        print("PASS: All inputs reconstructed correctly.")
        print("PASS: All token IDs are valid.")
    else:
        print("FAIL: Validation errors detected.")

    print("\nV4 validation completed.")


if __name__ == "__main__":
    main()
