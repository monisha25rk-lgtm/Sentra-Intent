
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "tokenizer")
)

from hybrid_bpe import (
    load_hybrid_tokenizer,
    encode_hybrid
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "intent"
    / "sentra_test1_clean.jsonl"
)

WORD_MERGES_PATH = (
    PROJECT_ROOT
    / "tokenizer"
    / "merges_word_v3.txt"
)

PHRASE_MERGES_PATH = (
    PROJECT_ROOT
    / "tokenizer"
    / "merges_phrase_v4.txt"
)


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

    print("\n===== SENTRA V4 TOKENIZATION INSPECTION =====")
    print("Word merges:", len(word_merges))
    print("Phrase merges:", len(phrase_merges))
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

            token_ids = encode_hybrid(
                prompt,
                word_merges,
                phrase_merges
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
