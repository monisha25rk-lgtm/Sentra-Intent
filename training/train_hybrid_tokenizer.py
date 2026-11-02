
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "tokenizer")
)

from word_bpe import load_word_merges
from hybrid_bpe import train_phrase_merges


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "intent"
    / "sentra_train_clean.jsonl"
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

MAX_PHRASE_MERGES = 512


def load_training_texts():

    texts = []

    print("Loading SENTRA training dataset...")

    with DATA_PATH.open("r", encoding="utf-8") as file:

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

            texts.append(model_input)

    print("Training records:", len(texts))

    return texts


def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    if not WORD_MERGES_PATH.exists():
        raise FileNotFoundError(WORD_MERGES_PATH)

    if PHRASE_MERGES_PATH.exists():
        raise FileExistsError(
            "V4 phrase merges already exist. "
            "Refusing to overwrite."
        )

    texts = load_training_texts()

    word_merges = load_word_merges(
        WORD_MERGES_PATH
    )

    print("\n===== SENTRA HYBRID TOKENIZER V4 =====")
    print("V3 word merges:", len(word_merges))
    print("Maximum phrase merges:", MAX_PHRASE_MERGES)

    phrase_merges = train_phrase_merges(
        texts=texts,
        word_merges=word_merges,
        phrase_merges_path=PHRASE_MERGES_PATH,
        max_phrase_merges=MAX_PHRASE_MERGES
    )

    final_vocab_size = (
        256
        + len(word_merges)
        + len(phrase_merges)
    )

    print("\n===== V4 TRAINING SUMMARY =====")
    print("Training records:", len(texts))
    print("Word-level merges:", len(word_merges))
    print("Phrase-level merges:", len(phrase_merges))
    print("Final vocabulary size:", final_vocab_size)
    print("Saved phrase merges:", PHRASE_MERGES_PATH)


if __name__ == "__main__":
    main()
