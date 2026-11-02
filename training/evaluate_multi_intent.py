
import sys
import json
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "training"))

from sentra_extractor import SentraExtractor
from train_sentra import SentraDataset, collate_batch
from torch.utils.data import DataLoader

DEVICE = torch.device("cpu")
PAD_ID = 1988
VOCAB_SIZE = 1989
THRESHOLD = 0.5

TEST_FILE = ROOT / "data" / "intent" / "encoded" / "test1_encoded.jsonl"
CHECKPOINT_FILE = ROOT / "checkpoints" / "sentra_best.pt"
MAPPING_FILE = ROOT / "configs" / "label_mappings.json"

FIELDS = ["intent_labels"]

with MAPPING_FILE.open("r", encoding="utf-8") as file:
    mappings = json.load(file)

id_to_label = {
    int(class_id): label
    for label, class_id in mappings["intent_labels"].items()
}

dataset = SentraDataset(TEST_FILE)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_batch
)

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE,
    weights_only=True
)

model = SentraExtractor(
    vocab_size=VOCAB_SIZE,
    padding_idx=PAD_ID
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_actual = []
all_predicted = []

exact_matches = 0
multi_total = 0
multi_exact_matches = 0

with torch.no_grad():

    for input_ids, lengths, targets in loader:

        input_ids = input_ids.to(DEVICE)

        outputs = model(input_ids, lengths)

        probabilities = torch.sigmoid(
            outputs["intent_labels"]
        )

        predictions = (
            probabilities >= THRESHOLD
        ).int()

        actual = targets["intent_labels"].int()

        for i in range(len(predictions)):

            actual_ids = set(
                torch.where(actual[i] == 1)[0].tolist()
            )

            predicted_ids = set(
                torch.where(predictions[i] == 1)[0].tolist()
            )

            all_actual.append(actual_ids)
            all_predicted.append(predicted_ids)

            if actual_ids == predicted_ids:
                exact_matches += 1

            if len(actual_ids) > 1:

                multi_total += 1

                if actual_ids == predicted_ids:
                    multi_exact_matches += 1


# --------------------------------------------------
# OVERALL EXACT MATCH
# --------------------------------------------------

total = len(all_actual)

print("\n===== INTENT EVALUATION =====")
print("Test records:", total)

print(
    "Overall exact match:",
    round(exact_matches / total, 4)
)


# --------------------------------------------------
# MULTI-INTENT EXACT MATCH
# --------------------------------------------------

print("\n===== MULTI-INTENT RESULTS =====")
print("Multi-intent records:", multi_total)

if multi_total:

    print(
        "Multi-intent exact match:",
        round(multi_exact_matches / multi_total, 4)
    )


# --------------------------------------------------
# PER-INTENT METRICS
# --------------------------------------------------

actual_matrix = []
predicted_matrix = []

for class_id in range(len(id_to_label)):

    actual_matrix.append([
        int(class_id in labels)
        for labels in all_actual
    ])

    predicted_matrix.append([
        int(class_id in labels)
        for labels in all_predicted
    ])

actual_matrix = torch.tensor(actual_matrix).T.numpy()
predicted_matrix = torch.tensor(predicted_matrix).T.numpy()

precision, recall, f1, support = (
    precision_recall_fscore_support(
        actual_matrix,
        predicted_matrix,
        average=None,
        zero_division=0
    )
)

print("\n===== PER-INTENT METRICS =====")

for class_id, label in id_to_label.items():

    print(
        f"{label:20} "
        f"P={precision[class_id]:.4f} "
        f"R={recall[class_id]:.4f} "
        f"F1={f1[class_id]:.4f} "
        f"Support={support[class_id]}"
    )

print("\nEvaluation completed.")
