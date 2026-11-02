
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "tokenizer")
)

from word_bpe import (
    load_tokenizer,
    encode
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "intent"
    / "sentra_test1_clean.jsonl"
)

MERGES_PATH = (
    PROJECT_ROOT
    / "tokenizer"
    / "merges_word_v3.txt"
)


def main():

    merges, vocabulary, vocab_size = load_tokenizer(
        MERGES_PATH
    )

    print("\n===== SENTRA V3 TOKENIZATION INSPECTION =====")
    print("Vocabulary size:", vocab_size)

    with DATA_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        for index, line in enumerate(file):

            if index >= 10:
                break

            record = json.loads(line)

            prompt = record["prompt"]

            token_ids = encode(
                prompt,
                merges
            )

            print("\n------------------------------------")
            print("Prompt:", prompt)
            print("Token IDs:", token_ids)
            print("Token count:", len(token_ids))

            print("Decoded token pieces:")

            for token_id in token_ids:

                token_piece = vocabulary[token_id].decode(
                    "utf-8",
                    errors="replace"
                )

                print(
                    f"  {token_id}: {repr(token_piece)}"
                )


if __name__ == "__main__":
    main()
