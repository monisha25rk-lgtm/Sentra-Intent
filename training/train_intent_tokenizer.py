
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer.byte_bpe import (
    train_tokenizer,
    load_tokenizer_config
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "intent"
    / "sentra_train_clean.jsonl"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "tokenizer_config.yaml"
)


def build_training_corpus():

    print("\nLoading SENTRA training dataset...")

    corpus = []

    with DATA_PATH.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            record = json.loads(line)

            prompt = record["prompt"]

            context = json.dumps(
                record["conversation_context"],
                ensure_ascii=False
            )

            text = (
                "[CONTEXT] "
                + context
                + " [PROMPT] "
                + prompt
                + "\n"
            )

            corpus.append(text)

    corpus_text = "".join(corpus)

    corpus_path = (
        PROJECT_ROOT
        / "data"
        / "intent"
        / "tokenizer_corpus.txt"
    )

    corpus_path.write_text(
        corpus_text,
        encoding="utf-8"
    )

    print("Training records:", len(corpus))
    print("Corpus saved:", corpus_path)

    return corpus_path


def train_intent_tokenizer():

    config = load_tokenizer_config()

    base_vocab_size = config[
        "tokenizer"
    ]["base_vocab_size"]

    merges_path = (
        PROJECT_ROOT
        / "tokenizer"
        / "merges_intent_v1.txt"
    )

    if merges_path.exists():
        raise FileExistsError(
            "Intent tokenizer already exists. "
            "Refusing to overwrite it."
        )

    corpus_path = build_training_corpus()

    print("\nTraining SENTRA Intent Tokenizer...")
    print("Base vocabulary:", base_vocab_size)

    merges = train_tokenizer(
        corpus_path=corpus_path,
        merges_path=merges_path,
        base_vocab_size=base_vocab_size
    )

    print("\n===== TOKENIZER TRAINING COMPLETE =====")
    print("Learned merges:", len(merges))
    print("Vocabulary size:", base_vocab_size + len(merges))
    print("Tokenizer saved:", merges_path)


if __name__ == "__main__":
    train_intent_tokenizer()
