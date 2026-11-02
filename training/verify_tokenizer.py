
import sys
from pathlib import Path

# Add project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer.byte_bpe import (
    load_tokenizer,
    encode,
    decode
)

def verify_tokenizer():

    print("\n========== SENTRA TOKENIZER VERIFICATION ==========\n")

    # 1. Check required files
    config_path = PROJECT_ROOT / "configs" / "tokenizer_config.yaml"
    tokenizer_path = PROJECT_ROOT / "tokenizer" / "byte_bpe.py"
    merges_path = PROJECT_ROOT / "tokenizer" / "merges.txt"

    files = {
        "Tokenizer configuration": config_path,
        "Tokenizer implementation": tokenizer_path,
        "Learned merges": merges_path
    }

    for name, path in files.items():
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{name}: {status}")

        if not path.exists():
            raise FileNotFoundError(path)

    # 2. Count merge rules
    merges_count = sum(
        1 for line in merges_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )

    print(f"\nLearned merge rules: {merges_count}")

    # 3. Load existing tokenizer
    merges, vocabulary, vocabulary_size, config = load_tokenizer()

    print(f"Tokenizer version: {config['tokenizer_version']}")
    print(f"Tokenizer type: {config['tokenizer']['type']}")
    print(f"Base vocabulary size: {config['tokenizer']['base_vocab_size']}")
    print(f"Vocabulary size: {vocabulary_size}")

    # 4. Test different kinds of input
    test_sentences = [
        "Summarize the API key rotation policy for Finance.",
        "Extract customer information from the database.",
        "Upload the report to a public website.",
        "Do not reveal confidential credentials."
    ]

    print("\n========== ENCODING AND DECODING TEST ==========\n")

    for sentence in test_sentences:

        token_ids = encode(sentence, merges)
        decoded_text = decode(token_ids, vocabulary)

        print("Original:", sentence)
        print("Token IDs:", token_ids)
        print("Token count:", len(token_ids))
        print("Decoded:", decoded_text)
        print("Round-trip:", decoded_text == sentence)
        print("-" * 55)

        if decoded_text != sentence:
            raise ValueError("Tokenizer round-trip test failed!")

    print("\nALL TOKENIZER TESTS PASSED!")

if __name__ == "__main__":
    verify_tokenizer()
