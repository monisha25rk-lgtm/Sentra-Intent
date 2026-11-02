
import re
from pathlib import Path
from collections import Counter


BASE_VOCAB_SIZE = 256

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ==========================================
# WORD-BOUNDARY PRE-TOKENIZATION
# ==========================================

def pre_tokenize(text):
    """
    Split text into words, whitespace groups,
    and punctuation groups.

    BPE merges cannot cross these boundaries.
    """

    return re.findall(
        r"\s+|[^\W_]+|_+|[^\w\s]+",
        text,
        flags=re.UNICODE
    )


# ==========================================
# APPLY ONE MERGE
# ==========================================

def apply_merge(sequence, pair, new_token_id):

    result = []
    i = 0

    while i < len(sequence):

        if (
            i < len(sequence) - 1
            and sequence[i] == pair[0]
            and sequence[i + 1] == pair[1]
        ):
            result.append(new_token_id)
            i += 2

        else:
            result.append(sequence[i])
            i += 1

    return result


# ==========================================
# TRAIN WORD-BOUNDARY BPE
# ==========================================

def train_word_bpe(
    texts,
    merges_path,
    max_merges=2048
):

    merges_path = Path(merges_path)

    if merges_path.exists():
        raise FileExistsError(
            f"Tokenizer already exists: {merges_path}"
        )

    sequences = []

    for text in texts:

        pieces = pre_tokenize(text)

        for piece in pieces:

            byte_sequence = list(
                piece.encode("utf-8")
            )

            if byte_sequence:
                sequences.append(byte_sequence)

    print("Pre-tokenized sequences:", len(sequences))

    learned_merges = []
    next_token_id = BASE_VOCAB_SIZE

    while len(learned_merges) < max_merges:

        pair_counts = Counter()

        for sequence in sequences:
            pair_counts.update(
                zip(sequence, sequence[1:])
            )

        if not pair_counts:
            break

        most_common_pair, count = (
            pair_counts.most_common(1)[0]
        )

        if count < 2:
            break

        new_token_id = next_token_id

        sequences = [
            apply_merge(
                sequence,
                most_common_pair,
                new_token_id
            )
            for sequence in sequences
        ]

        learned_merges.append(
            (
                most_common_pair[0],
                most_common_pair[1],
                new_token_id
            )
        )

        next_token_id += 1

        if len(learned_merges) % 100 == 0:
            print(
                "Learned merges:",
                len(learned_merges)
            )

    merges_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with merges_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        for first, second, token_id in learned_merges:
            file.write(
                f"{first} {second} {token_id}\n"
            )

    print("\nTokenizer training completed.")
    print("Learned merges:", len(learned_merges))
    print(
        "Vocabulary size:",
        BASE_VOCAB_SIZE + len(learned_merges)
    )
    print("Saved to:", merges_path)

    return learned_merges


# ==========================================
# LOAD MERGES
# ==========================================

def load_word_merges(merges_path):

    merges = []

    with Path(merges_path).open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            first, second, token_id = map(
                int,
                line.split()
            )

            merges.append(
                (first, second, token_id)
            )

    return merges


# ==========================================
# BUILD VOCABULARY
# ==========================================

def build_vocabulary(merges):

    vocabulary = {
        i: bytes([i])
        for i in range(BASE_VOCAB_SIZE)
    }

    for first, second, token_id in merges:

        if first not in vocabulary:
            raise ValueError(
                f"Unknown token ID: {first}"
            )

        if second not in vocabulary:
            raise ValueError(
                f"Unknown token ID: {second}"
            )

        vocabulary[token_id] = (
            vocabulary[first] + vocabulary[second]
        )

    return vocabulary


# ==========================================
# ENCODE
# ==========================================

def encode(text, merges):

    token_ids = []

    for piece in pre_tokenize(text):

        sequence = list(piece.encode("utf-8"))

        for first, second, token_id in merges:

            sequence = apply_merge(
                sequence,
                (first, second),
                token_id
            )

        token_ids.extend(sequence)

    return token_ids


# ==========================================
# DECODE
# ==========================================

def decode(token_ids, vocabulary):

    byte_sequence = b""

    for token_id in token_ids:

        if token_id not in vocabulary:
            raise ValueError(
                f"Unknown token ID: {token_id}"
            )

        byte_sequence += vocabulary[token_id]

    return byte_sequence.decode("utf-8")


# ==========================================
# LOAD TRAINED TOKENIZER
# ==========================================

def load_tokenizer(merges_path):

    merges = load_word_merges(merges_path)

    vocabulary = build_vocabulary(merges)

    vocabulary_size = (
        BASE_VOCAB_SIZE + len(merges)
    )

    return merges, vocabulary, vocabulary_size
