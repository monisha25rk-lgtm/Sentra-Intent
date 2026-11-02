
from pathlib import Path
from collections import Counter

from word_bpe import (
    BASE_VOCAB_SIZE,
    encode as encode_word,
    build_vocabulary as build_word_vocabulary
)


# ==========================================
# APPLY PHRASE MERGE
# ==========================================

def apply_phrase_merge(sequence, pair, new_token_id):

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
# TRAIN PHRASE LAYER
# ==========================================

def train_phrase_merges(
    texts,
    word_merges,
    phrase_merges_path,
    max_phrase_merges=512
):

    phrase_merges_path = Path(phrase_merges_path)

    if phrase_merges_path.exists():
        raise FileExistsError(
            f"Phrase tokenizer already exists: "
            f"{phrase_merges_path}"
        )

    print("Encoding training text using V3...")

    sequences = [
        encode_word(text, word_merges)
        for text in texts
    ]

    next_token_id = (
        BASE_VOCAB_SIZE + len(word_merges)
    )

    learned_merges = []

    print("Starting phrase-level BPE...")

    while len(learned_merges) < max_phrase_merges:

        pair_counts = Counter()

        for sequence in sequences:
            pair_counts.update(
                zip(sequence, sequence[1:])
            )

        if not pair_counts:
            break

        pair, count = pair_counts.most_common(1)[0]

        if count < 2:
            break

        new_token_id = next_token_id

        sequences = [
            apply_phrase_merge(
                sequence,
                pair,
                new_token_id
            )
            for sequence in sequences
        ]

        learned_merges.append(
            (pair[0], pair[1], new_token_id)
        )

        next_token_id += 1

        if len(learned_merges) % 50 == 0:
            print(
                "Phrase merges learned:",
                len(learned_merges)
            )

    phrase_merges_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with phrase_merges_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        for first, second, token_id in learned_merges:
            file.write(
                f"{first} {second} {token_id}\n"
            )

    print("\nPhrase training completed.")
    print("Phrase merges:", len(learned_merges))
    print(
        "Final vocabulary size:",
        BASE_VOCAB_SIZE
        + len(word_merges)
        + len(learned_merges)
    )

    return learned_merges


# ==========================================
# LOAD PHRASE MERGES
# ==========================================

def load_phrase_merges(path):

    merges = []

    with Path(path).open(
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
# BUILD HYBRID VOCABULARY
# ==========================================

def build_hybrid_vocabulary(
    word_merges,
    phrase_merges
):

    vocabulary = build_word_vocabulary(
        word_merges
    )

    for first, second, token_id in phrase_merges:

        if first not in vocabulary:
            raise ValueError(
                f"Unknown phrase merge token: {first}"
            )

        if second not in vocabulary:
            raise ValueError(
                f"Unknown phrase merge token: {second}"
            )

        vocabulary[token_id] = (
            vocabulary[first] + vocabulary[second]
        )

    return vocabulary


# ==========================================
# LOAD HYBRID TOKENIZER
# ==========================================

def load_hybrid_tokenizer(
    word_merges_path,
    phrase_merges_path
):

    from word_bpe import load_word_merges

    word_merges = load_word_merges(
        word_merges_path
    )

    phrase_merges = load_phrase_merges(
        phrase_merges_path
    )

    vocabulary = build_hybrid_vocabulary(
        word_merges,
        phrase_merges
    )

    vocabulary_size = (
        BASE_VOCAB_SIZE
        + len(word_merges)
        + len(phrase_merges)
    )

    return (
        word_merges,
        phrase_merges,
        vocabulary,
        vocabulary_size
    )


# ==========================================
# HYBRID ENCODING
# ==========================================

def encode_hybrid(
    text,
    word_merges,
    phrase_merges
):

    token_ids = encode_word(
        text,
        word_merges
    )

    for first, second, token_id in phrase_merges:

        token_ids = apply_phrase_merge(
            token_ids,
            (first, second),
            token_id
        )

    return token_ids


# ==========================================
# HYBRID DECODING
# ==========================================

def decode_hybrid(token_ids, vocabulary):

    byte_sequence = b""

    for token_id in token_ids:

        if token_id not in vocabulary:
            raise ValueError(
                f"Unknown token ID: {token_id}"
            )

        byte_sequence += vocabulary[token_id]

    return byte_sequence.decode("utf-8")
