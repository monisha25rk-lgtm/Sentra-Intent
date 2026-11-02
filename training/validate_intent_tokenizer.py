
import sys
import json
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer.byte_bpe import load_tokenizer, encode, decode

DATA_DIR = PROJECT_ROOT / "data" / "intent"

DATA_FILES = [
    "sentra_train_clean.jsonl",
    "sentra_test1_clean.jsonl",
    "sentra_test2_clean.jsonl"
]


def validate_tokenizer():

    print("\n========== SENTRA TOKENIZER V2 VALIDATION ==========\n")

    # Load the tokenizer specified in tokenizer_config.yaml
    merges, vocabulary, vocab_size, config = load_tokenizer()

    print("Tokenizer version:", config["tokenizer_version"])
    print("Tokenizer type:", config["tokenizer"]["type"])
    print("Learned merges:", len(merges))
    print("Vocabulary size:", vocab_size)

    total_records = 0
    total_tokens = 0
    total_characters = 0
    failed_roundtrips = 0
    invalid_token_ids = 0
    all_lengths = []

    for filename in DATA_FILES:

        file_path = DATA_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        record_count = 0
        split_tokens = 0
        split_characters = 0
        split_lengths = []

        with file_path.open("r", encoding="utf-8") as file:

            for line_number, line in enumerate(file, start=1):

                if not line.strip():
                    continue

                record = json.loads(line)

                prompt = record["prompt"]

                context = json.dumps(
                    record["conversation_context"],
                    ensure_ascii=False
                )

                # Use the same input structure used during training
                model_input = (
                    "[CONTEXT] "
                    + context
                    + " [PROMPT] "
                    + prompt
                )

                token_ids = encode(model_input, merges)

                # Check exact reconstruction
                reconstructed = decode(token_ids, vocabulary)

                if reconstructed != model_input:
                    failed_roundtrips += 1

                # Check every token ID
                for token_id in token_ids:
                    if token_id not in vocabulary:
                        invalid_token_ids += 1

                record_count += 1
                split_tokens += len(token_ids)
                split_characters += len(model_input)
                split_lengths.append(len(token_ids))

        total_records += record_count
        total_tokens += split_tokens
        total_characters += split_characters
        all_lengths.extend(split_lengths)

        print("\n----------------------------------------")
        print("Dataset:", filename)
        print("Records:", record_count)
        print("Total tokens:", split_tokens)
        print("Average tokens per record:",
              round(mean(split_lengths), 2))
        print("Maximum tokens in one record:",
              max(split_lengths))
        print("Minimum tokens in one record:",
              min(split_lengths))

    print("\n========== OVERALL RESULTS ==========")

    print("Total records:", total_records)
    print("Total tokens:", total_tokens)
    print("Total characters:", total_characters)

    print("Failed round trips:", failed_roundtrips)
    print("Invalid token IDs:", invalid_token_ids)

    print("Average tokens per record:",
          round(mean(all_lengths), 2))

    print("Maximum tokens per record:",
          max(all_lengths))

    print("Average characters per token:",
          round(total_characters / total_tokens, 3)
          if total_tokens else 0)

    # 95th percentile token length
    sorted_lengths = sorted(all_lengths)
    p95_index = int(0.95 * (len(sorted_lengths) - 1))

    print("95th percentile token length:",
          sorted_lengths[p95_index])

    print("\n========== VALIDATION STATUS ==========")

    if failed_roundtrips == 0 and invalid_token_ids == 0:
        print("PASS: All inputs reconstructed correctly.")
        print("PASS: All token IDs belong to the vocabulary.")
    else:
        print("FAIL: Validation errors detected.")

    print("\nTokenizer validation completed.")


if __name__ == "__main__":
    validate_tokenizer()
