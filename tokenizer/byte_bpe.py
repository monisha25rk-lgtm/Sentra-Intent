
from pathlib import Path
from collections import Counter
import yaml


# ========================================
# PROJECT ROOT
# ========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ========================================
# TOKENIZER CONFIGURATION
# ========================================

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "tokenizer_config.yaml"
)


def load_tokenizer_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return yaml.safe_load(file)


# ========================================
# APPLY ONE MERGE
# ========================================

def apply_merge(
    data,
    pair,
    new_token_id
):

    merged_data = []

    i = 0

    while i < len(data):

        if (
            i < len(data) - 1
            and data[i] == pair[0]
            and data[i + 1] == pair[1]
        ):

            merged_data.append(
                new_token_id
            )

            i += 2

        else:

            merged_data.append(
                data[i]
            )

            i += 1

    return merged_data


# ========================================
# LOAD LEARNED MERGES
# ========================================

def load_merges(
    merges_path
):

    merges_path = Path(
        merges_path
    )

    learned_merges = []

    if not merges_path.exists():

        return learned_merges

    lines = merges_path.read_text(
        encoding="utf-8"
    ).splitlines()

    for line in lines:

        if not line.strip():
            continue

        first, second, token_id = map(
            int,
            line.split()
        )

        learned_merges.append(
            (
                first,
                second,
                token_id
            )
        )

    return learned_merges


# ========================================
# GET VOCABULARY SIZE
# ========================================

def get_vocabulary_size(
    merges,
    base_vocab_size
):

    return (
        base_vocab_size
        + len(merges)
    )


# ========================================
# TRAIN BYTE-LEVEL BPE
# ========================================
#
# IMPORTANT:
#
# This function creates a tokenizer from
# scratch.
#
# It NEVER loads an existing tokenizer
# and NEVER appends new merges to an
# existing tokenizer.
#
# Therefore:
#
# Tokenizer V1 remains immutable once
# created.
#
# ========================================

def train_tokenizer(
    corpus_path,
    merges_path,
    base_vocab_size
):

    corpus_path = Path(
        corpus_path
    )

    merges_path = Path(
        merges_path
    )

    # --------------------------------
    # SAFETY CHECK
    # --------------------------------
    #
    # Never overwrite an existing
    # tokenizer accidentally.
    #
    if merges_path.exists():

        raise FileExistsError(
            f"Tokenizer already exists at: "
            f"{merges_path}\n"
            f"Existing tokenizer versions must "
            f"not be modified."
        )

    # --------------------------------
    # LOAD CORPUS
    # --------------------------------

    text = corpus_path.read_text(
        encoding="utf-8"
    )

    # --------------------------------
    # CONVERT TEXT TO UTF-8 BYTES
    # --------------------------------

    data = list(
        text.encode("utf-8")
    )

    print(
        "Initial number of symbols:",
        len(data)
    )

    # --------------------------------
    # LEARN MERGES FROM SCRATCH
    # --------------------------------

    next_token_id = (
        base_vocab_size
    )

    learned_merges = []

    while True:

        # Count adjacent token pairs.

        pair_counts = Counter(
            zip(
                data,
                data[1:]
            )
        )

        # No pairs remain.

        if not pair_counts:

            break

        # Find the most frequent pair.

        most_common_pair, count = (
            pair_counts.most_common(1)[0]
        )

        # A merge is useful only if the
        # pair occurs at least twice.

        if count < 2:

            break

        # Assign a new vocabulary ID.

        new_token_id = (
            next_token_id
        )

        # Apply the merge to the
        # current token stream.

        data = apply_merge(
            data,
            most_common_pair,
            new_token_id
        )

        # Store the learned merge.

        learned_merges.append(
            (
                most_common_pair[0],
                most_common_pair[1],
                new_token_id
            )
        )

        print(
            f"Merge #{len(learned_merges)}: "
            f"{most_common_pair} -> "
            f"{new_token_id} "
            f"({count} occurrences)"
        )

        next_token_id += 1

    # --------------------------------
    # SAVE TOKENIZER
    # --------------------------------

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
                f"{first} "
                f"{second} "
                f"{token_id}\n"
            )

    # --------------------------------
    # VOCABULARY SIZE
    # --------------------------------

    vocabulary_size = (
        get_vocabulary_size(
            learned_merges,
            base_vocab_size
        )
    )

    print()

    print(
        "Tokenizer training completed."
    )

    print(
        "Learned merges:",
        len(learned_merges)
    )

    print(
        "Tokenizer vocabulary size:",
        vocabulary_size
    )

    print(
        "Saved tokenizer to:",
        merges_path
    )

    return learned_merges


# ========================================
# BUILD TOKEN VOCABULARY
# ========================================

def build_token_vocabulary(
    merges,
    base_vocab_size
):

    vocabulary = {
        i: bytes([i])
        for i in range(
            base_vocab_size
        )
    }

    for first, second, token_id in merges:

        if first not in vocabulary:

            raise ValueError(
                f"Unknown token ID in merge: "
                f"{first}"
            )

        if second not in vocabulary:

            raise ValueError(
                f"Unknown token ID in merge: "
                f"{second}"
            )

        vocabulary[token_id] = (
            vocabulary[first]
            + vocabulary[second]
        )

    return vocabulary


# ========================================
# ENCODE
# ========================================

def encode(
    text,
    merges
):

    data = list(
        text.encode("utf-8")
    )

    for first, second, token_id in merges:

        data = apply_merge(
            data,
            (
                first,
                second
            ),
            token_id
        )

    return data


# ========================================
# DECODE
# ========================================

def decode(
    token_ids,
    vocabulary
):

    byte_sequence = b""

    for token_id in token_ids:

        if token_id not in vocabulary:

            raise ValueError(
                f"Unknown token ID: "
                f"{token_id}"
            )

        byte_sequence += (
            vocabulary[token_id]
        )

    return byte_sequence.decode(
        "utf-8"
    )


# ========================================
# LOAD EXISTING TOKENIZER
# ========================================
#
# This function ONLY loads.
#
# It NEVER trains.
# It NEVER modifies merges.txt.
#
# This is what Model V1, V2, V3...
# will use.
#
# ========================================

def load_tokenizer():

    config = load_tokenizer_config()

    base_vocab_size = config[
        "tokenizer"
    ][
        "base_vocab_size"
    ]

    merges_path = (
        PROJECT_ROOT
        / config["tokenizer"]["merges_path"]
    )

    if not merges_path.exists():

        raise FileNotFoundError(
            f"Tokenizer does not exist at: "
            f"{merges_path}\n"
            f"Create Tokenizer V1 first."
        )

    merges = load_merges(
        merges_path
    )

    vocabulary = build_token_vocabulary(
        merges,
        base_vocab_size
    )

    vocabulary_size = (
        get_vocabulary_size(
            merges,
            base_vocab_size
        )
    )

    return (
        merges,
        vocabulary,
        vocabulary_size,
        config
    )


# ========================================
# CREATE TOKENIZER V1
# ========================================

def create_tokenizer():

    config = load_tokenizer_config()

    corpus_path = (
        PROJECT_ROOT
        / config["source"]["path"]
    )

    merges_path = (
        PROJECT_ROOT
        / config["tokenizer"]["merges_path"]
    )

    base_vocab_size = config[
        "tokenizer"
    ][
        "base_vocab_size"
    ]

    # --------------------------------
    # NEVER MODIFY EXISTING TOKENIZER
    # --------------------------------

    if merges_path.exists():

        print(
            "Tokenizer already exists."
        )

        print(
            "Existing tokenizer will NOT "
            "be modified."
        )

        print(
            "Loading existing tokenizer..."
        )

        return load_tokenizer()

    # --------------------------------
    # CREATE NEW TOKENIZER
    # --------------------------------

    merges = train_tokenizer(
        corpus_path=corpus_path,
        merges_path=merges_path,
        base_vocab_size=base_vocab_size
    )

    vocabulary = build_token_vocabulary(
        merges,
        base_vocab_size
    )

    vocabulary_size = (
        get_vocabulary_size(
            merges,
            base_vocab_size
        )
    )

    return (
        merges,
        vocabulary,
        vocabulary_size,
        config
    )


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":

    (
        merges,
        vocabulary,
        vocabulary_size,
        config
    ) = create_tokenizer()

    # --------------------------------
    # TOKENIZER INFORMATION
    # --------------------------------

    print()

    print(
        "========================================"
    )

    print(
        "SENTRA TOKENIZER"
    )

    print(
        "========================================"
    )

    print(
        "Tokenizer version:",
        config["tokenizer_version"]
    )

    print(
        "Tokenizer type:",
        config["tokenizer"]["type"]
    )

    print(
        "Base vocabulary size:",
        config["tokenizer"]["base_vocab_size"]
    )

    print(
        "Learned merges:",
        len(merges)
    )

    print(
        "Vocabulary size:",
        vocabulary_size
    )

    # --------------------------------
    # ROUND-TRIP VALIDATION
    # --------------------------------

    test_sentence = (
        "The server rejected an invalid request."
    )

    encoded = encode(
        test_sentence,
        merges
    )

    decoded = decode(
        encoded,
        vocabulary
    )

    print()

    print(
        "Test sentence:"
    )

    print(
        test_sentence
    )

    print()

    print(
        "Encoded token IDs:"
    )

    print(
        encoded
    )

    print()

    print(
        "Number of tokens:"
    )

    print(
        len(encoded)
    )

    print()

    print(
        "Decoded text:"
    )

    print(
        decoded
    )

    print()

    print(
        "Round-trip successful:"
    )

    print(
        decoded == test_sentence
    )
