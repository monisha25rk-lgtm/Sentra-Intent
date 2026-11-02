
import sys
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "training"))

from sentra_extractor import SentraExtractor
from sentra_losses import SentraMultiTaskLoss


MULTI_FIELDS = [
    "intent_labels",
    "action",
    "data_type"
]

SINGLE_FIELDS = [
    "scenario_family",
    "target",
    "destination",
    "authorization",
    "context_dependency"
]

PAD_ID = 1988


def load_sample_batch():

    path = (
        ROOT / "data" / "intent" /
        "encoded" / "train_encoded.jsonl"
    )

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

            if len(records) == 4:
                break

    lengths = torch.tensor(
        [len(record["input_ids"]) for record in records],
        dtype=torch.long
    )

    max_length = max(lengths).item()

    input_ids = []

    for record in records:

        ids = record["input_ids"]

        padded = ids + [PAD_ID] * (max_length - len(ids))

        input_ids.append(padded)

    input_ids = torch.tensor(input_ids, dtype=torch.long)

    targets = {}

    # Convert multi-label IDs into multi-hot vectors.
    for field in MULTI_FIELDS:

        number_of_labels = {
            "intent_labels": 9,
            "action": 9,
            "data_type": 14
        }[field]

        target_tensor = torch.zeros(
            len(records),
            number_of_labels
        )

        for i, record in enumerate(records):

            for label_id in record["labels"][field]:
                target_tensor[i, label_id] = 1.0

        targets[field] = target_tensor

    # Single-label targets remain integer IDs.
    for field in SINGLE_FIELDS:

        targets[field] = torch.tensor(
            [
                record["labels"][field]
                for record in records
            ],
            dtype=torch.long
        )

    return input_ids, lengths, targets


def main():

    print("===== SENTRA LOSS TEST =====")

    input_ids, lengths, targets = load_sample_batch()

    model = SentraExtractor()
    criterion = SentraMultiTaskLoss()

    model.train()

    outputs = model(input_ids, lengths)

    total_loss, individual_losses = criterion(
        outputs,
        targets
    )

    print("\n===== INDIVIDUAL LOSSES =====")

    for field, loss in individual_losses.items():

        print(
            f"{field:25} "
            f"{loss.item():.6f}"
        )

        assert torch.isfinite(loss), (
            f"Invalid loss for {field}"
        )

    print("\nTotal loss:", total_loss.item())

    assert torch.isfinite(total_loss)

    # Backpropagation test.
    model.zero_grad()

    total_loss.backward()

    print("\n===== GRADIENT CHECK =====")

    for field, head in model.heads.items():

        gradient = head.weight.grad

        passed = (
            gradient is not None
            and torch.isfinite(gradient).all().item()
            and gradient.abs().sum().item() > 0
        )

        print(
            f"{field:25} "
            f"{'PASS' if passed else 'FAIL'}"
        )

        assert passed, (
            f"Gradient check failed for {field}"
        )

    print("\nALL LOSS AND GRADIENT TESTS PASSED.")


if __name__ == "__main__":
    main()
