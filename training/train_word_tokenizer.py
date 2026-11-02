import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "tokenizer")
)

from word_bpe import train_word_bpe


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "intent"
    / "sentra_train_clean.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "tokenizer"
    / "merges_word_v3.txt"
)

MAX_MERGES = 2048


def load_training_texts():

    texts = []

    print("Loading SENTRA training dataset...")

    with DATA_PATH.open("r", encoding="utf-8") as file:

        for line in file:

            if not line.strip():
                continue

            record = json.loads(line)

            prompt = record["prompt"]

            context = json.dumps(
                record["conversation_context"],
                ensure_ascii=False
            )

            model_input = (
                "[CONTEXT] "
                + context
                + " [PROMPT] "
                + prompt
            )

            texts.append(model_input)

    print("Training records loaded:", len(texts))

    return texts


def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            "V3 tokenizer already exists. "
            "Refusing to overwrite it."
        )

    texts = load_training_texts()

    print("\nStarting SENTRA Word-Boundary BPE V3")
    print("Maximum merges:", MAX_MERGES)

    merges = train_word_bpe(
        texts=texts,
        merges_path=OUTPUT_PATH,
        max_merges=MAX_MERGES
    )

    print("\n========== V3 TRAINING SUMMARY ==========")
    print("Training records:", len(texts))
    print("Learned merges:", len(merges))
    print("Vocabulary size:", 256 + len(merges))
    print("Tokenizer file:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
