
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    ROOT / "data" / "intent" / "encoded" / "train_grouped_encoded.jsonl"
)

MAPPING_FILE = ROOT / "configs" / "label_mappings.json"


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


def calculate_training_weights():

    with MAPPING_FILE.open("r", encoding="utf-8") as file:
        mappings = json.load(file)

    total_records = 0

    positive_counts = {
        field: Counter()
        for field in MULTI_LABEL_FIELDS
    }

    authorization_counts = Counter()

    with TRAIN_FILE.open("r", encoding="utf-8") as file:

        for line in file:

            record = json.loads(line)
            labels = record["labels"]

            total_records += 1

            for field in MULTI_LABEL_FIELDS:
                positive_counts[field].update(labels[field])

            authorization_counts.update(
                [labels["authorization"]]
            )

    # Calculate positive weights for multi-label tasks.
    pos_weights = {}

    for field in MULTI_LABEL_FIELDS:

        number_of_labels = len(mappings[field])
        weights = []

        for label_id in range(number_of_labels):

            positives = positive_counts[field][label_id]
            negatives = total_records - positives

            if positives == 0:
                raise ValueError(
                    f"No positive training examples for "
                    f"{field}, label ID {label_id}"
                )

            weights.append(negatives / positives)

        pos_weights[field] = torch.tensor(
            weights,
            dtype=torch.float32
        )

    # Calculate balanced class weights for authorization.
    authorization_classes = len(mappings["authorization"])

    auth_weights = []

    for class_id in range(authorization_classes):

        count = authorization_counts[class_id]

        if count == 0:
            raise ValueError(
                f"No authorization examples for class {class_id}"
            )

        weight = total_records / (
            authorization_classes * count
        )

        auth_weights.append(weight)

    auth_weights = torch.tensor(
        auth_weights,
        dtype=torch.float32
    )

    return pos_weights, auth_weights


class SentraMultiTaskLoss(nn.Module):

    def __init__(self):

        super().__init__()

        pos_weights, auth_weights = calculate_training_weights()

        self.multi_label_losses = nn.ModuleDict({
            field: nn.BCEWithLogitsLoss(
                pos_weight=pos_weights[field]
            )
            for field in MULTI_LABEL_FIELDS
        })

        self.single_label_losses = nn.ModuleDict({
            field: nn.CrossEntropyLoss(
                weight=auth_weights if field == "authorization" else None
            )
            for field in SINGLE_LABEL_FIELDS
        })

    def forward(self, outputs, targets):

        individual_losses = {}

        # Multi-label classification losses.
        for field in MULTI_LABEL_FIELDS:

            individual_losses[field] = (
                self.multi_label_losses[field](
                    outputs[field],
                    targets[field].float()
                )
            )

        # Single-label classification losses.
        for field in SINGLE_LABEL_FIELDS:

            individual_losses[field] = (
                self.single_label_losses[field](
                    outputs[field],
                    targets[field].long()
                )
            )

        # Equal contribution from all eight tasks initially.
        total_loss = torch.stack(
            list(individual_losses.values())
        ).sum()

        return total_loss, individual_losses
