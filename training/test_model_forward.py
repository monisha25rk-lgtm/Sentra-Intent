
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from sentra_extractor import SentraExtractor


def main():

    print("===== SENTRA NEURAL NETWORK TEST =====")

    model = SentraExtractor()
    model.eval()

    print("Model initialized successfully.")

    PAD_ID = 1988

    input_ids = torch.tensor([
        [1540, 767, 1965, 1012, 1812, 956, 46, PAD_ID, PAD_ID, PAD_ID],
        [1519, 902, 1743, 1551, 1366, 32, 1367, 63, 46, 767]
    ], dtype=torch.long)

    lengths = torch.tensor([7, 10])

    with torch.no_grad():
        outputs = model(input_ids, lengths)

    expected_shapes = {
        "scenario_family": (2, 5),
        "intent_labels": (2, 9),
        "action": (2, 9),
        "target": (2, 56),
        "data_type": (2, 14),
        "destination": (2, 10),
        "authorization": (2, 3),
        "context_dependency": (2, 2)
    }

    print("\n===== PREDICTION OUTPUTS =====")

    for field, expected in expected_shapes.items():

        actual = tuple(outputs[field].shape)

        if actual != expected:
            raise ValueError(
                f"{field}: expected {expected}, got {actual}"
            )

        if not torch.isfinite(outputs[field]).all():
            raise ValueError(f"Invalid values in {field}")

        print(f"{field:25} {actual} PASS")

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("\nTotal parameters:", parameters)
    print("Vocabulary size:", model.embedding.num_embeddings)
    print("\nALL MODEL TESTS PASSED.")


if __name__ == "__main__":
    main()
