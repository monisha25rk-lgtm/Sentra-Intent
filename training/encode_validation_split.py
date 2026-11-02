
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "training"))

from encode_intent_dataset import (
    load_hybrid_tokenizer,
    load_label_mappings,
    encode_record
)

import json

INPUT_DIR = ROOT / "data" / "intent" / "split"
OUTPUT_DIR = ROOT / "data" / "intent" / "encoded"

WORD_MERGES = ROOT / "tokenizer" / "merges_word_v3.txt"
PHRASE_MERGES = ROOT / "tokenizer" / "merges_phrase_v4.txt"


def main():

    (
        word_merges,
        phrase_merges,
        vocabulary,
        vocabulary_size
    ) = load_hybrid_tokenizer(
        WORD_MERGES,
        PHRASE_MERGES
    )

    mappings = load_label_mappings()

    print("V4 vocabulary size:", vocabulary_size)

    for split_name in ["train", "validation"]:

        input_file = INPUT_DIR / f"{split_name}.jsonl"

        output_name = (
            "train_grouped_encoded.jsonl"
            if split_name == "train"
            else "validation_encoded.jsonl"
        )

        output_file = OUTPUT_DIR / output_name

        count = 0
        total_tokens = 0

        with input_file.open("r", encoding="utf-8") as source, \
             output_file.open("w", encoding="utf-8") as destination:

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
        print("Average tokens:", round(total_tokens / count, 2))
        print("Saved:", output_file)


if __name__ == "__main__":
    main()
