
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer.byte_bpe import load_tokenizer, encode, decode

DATA_DIR = PROJECT_ROOT / "data" / "intent"

DATA_FILES = [
    "sentra_train_clean.jsonl",
    "sentra_test1_clean.jsonl",
    "sentra_test2_clean.jsonl"
]

def main():

    print("\n===== SENTRA DATASET TOKENIZATION TEST =====")

    merges, vocabulary, vocab_size, config = load_tokenizer()

    print("Tokenizer loaded successfully")
    print("Vocabulary size:", vocab_size)

    total_records = 0
    total_tokens = 0
    total_characters = 0
    failed_roundtrips = 0

    for filename in DATA_FILES:

        file_path = DATA_DIR / filename
        records = 0
        tokens = 0
        characters = 0

        with file_path.open("r", encoding="utf-8") as file:

            for line in file:

                if not line.strip():
                    continue

                record = json.loads(line)

                prompt = record["prompt"]
                context = record["conversation_context"]

                # Test prompt
                prompt_ids = encode(prompt, merges)
                prompt_decoded = decode(prompt_ids, vocabulary)

                if prompt_decoded != prompt:
                    failed_roundtrips += 1

                # Test conversation context
                context_text = json.dumps(
                    context,
                    ensure_ascii=False
                )

                context_ids = encode(context_text, merges)
                context_decoded = decode(context_ids, vocabulary)

                if context_decoded != context_text:
                    failed_roundtrips += 1

                records += 1
                tokens += len(prompt_ids) + len(context_ids)
                characters += len(prompt) + len(context_text)

        total_records += records
        total_tokens += tokens
        total_characters += characters

        print(f"\nFile: {filename}")
        print("Records:", records)
        print("Tokens:", tokens)
        print("Characters:", characters)

    print("\n===== FINAL RESULTS =====")
    print("Total records:", total_records)
    print("Total tokens:", total_tokens)
    print("Total characters:", total_characters)
    print("Failed round trips:", failed_roundtrips)

    if total_characters:
        print(
            "Average characters per token:",
            round(total_characters / total_tokens, 3)
            if total_tokens else 0
        )

    print("\nDataset tokenization test completed.")

if __name__ == "__main__":
    main()
