
import sys
import json
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    hamming_loss,
    f1_score
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "training"))

from sentra_extractor import SentraExtractor
from train_sentra import (
    SentraDataset,
    collate_batch,
    BATCH_SIZE
)

PAD_ID = 1988
VOCAB_SIZE = 1989

CHECKPOINT = ROOT / "checkpoints" / "sentra_best.pt"
VALIDATION_FILE = ROOT / "data" / "intent" / "encoded" / "test2_encoded.jsonl"

DEVICE = torch.device("cpu")

MULTI_LABEL_FIELDS = [
    "intent_labels",
    "action",
    "data_type"
]

SINGLE_LABEL_FIELDS = [
    "scenario_family",
    "target",
    "destination",
    "authorization",
    "context_dependency"
]


def calculate_single_label_metrics(y_true, y_pred, field):
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(f"\n{field}")
    print("-" * 55)
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro Precision:   {precision:.4f}")
    print(f"Macro Recall:      {recall:.4f}")
    print(f"Macro F1:          {f1:.4f}")


def calculate_multilabel_metrics(y_true, y_pred, field):
    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    hamming = hamming_loss(y_true, y_pred)

    print(f"\n{field}")
    print("-" * 55)
    print(f"Micro F1:          {micro_f1:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")
    print(f"Hamming Loss:      {hamming:.4f}")


def main():
    print("=" * 65)
    print("SENTRA MODEL EVALUATION")
    print("=" * 65)

    dataset = SentraDataset(VALIDATION_FILE)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model = SentraExtractor(
        vocab_size=VOCAB_SIZE,
        padding_idx=PAD_ID
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\nCheckpoint epoch:", checkpoint["epoch"])
    print("Validation records:", len(dataset))

    all_predictions = {field: [] for field in MULTI_LABEL_FIELDS + SINGLE_LABEL_FIELDS}
    all_targets = {field: [] for field in MULTI_LABEL_FIELDS + SINGLE_LABEL_FIELDS}

    with torch.no_grad():
        for input_ids, lengths, targets in loader:
            input_ids = input_ids.to(DEVICE)

            outputs = model(input_ids, lengths)

            for field in MULTI_LABEL_FIELDS:
                probabilities = torch.sigmoid(outputs[field])
                predictions = (probabilities >= 0.5).int()

                all_predictions[field].extend(predictions.cpu().numpy())
                all_targets[field].extend(targets[field].cpu().numpy())

            for field in SINGLE_LABEL_FIELDS:
                predictions = torch.argmax(outputs[field], dim=1)

                all_predictions[field].extend(predictions.cpu().numpy())
                all_targets[field].extend(targets[field].cpu().numpy())

    print("\n" + "=" * 65)
    print("SINGLE-LABEL EVALUATION")
    print("=" * 65)

    for field in SINGLE_LABEL_FIELDS:
        calculate_single_label_metrics(
            np.array(all_targets[field]),
            np.array(all_predictions[field]),
            field
        )

    print("\n" + "=" * 65)
    print("MULTI-LABEL EVALUATION")
    print("=" * 65)

    for field in MULTI_LABEL_FIELDS:
        calculate_multilabel_metrics(
            np.array(all_targets[field]),
            np.array(all_predictions[field]),
            field
        )

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
