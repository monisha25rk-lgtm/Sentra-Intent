
import sys
import json
import torch
from pathlib import Path
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "training"))

from sentra_extractor import SentraExtractor
from train_sentra import SentraDataset, collate_batch

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DEVICE = torch.device("cpu")
PAD_ID = 1988
VOCAB_SIZE = 1989

TEST_FILE = (
    ROOT / "data" / "intent" / "encoded" / "test1_encoded.jsonl"
)

CHECKPOINT_FILE = ROOT / "checkpoints" / "sentra_best.pt"
MAPPING_FILE = ROOT / "configs" / "label_mappings.json"

# --------------------------------------------------
# LOAD MAPPINGS
# --------------------------------------------------

with MAPPING_FILE.open("r", encoding="utf-8") as file:
    mappings = json.load(file)

destination_mapping = mappings["destination"]

id_to_label = {
    int(class_id): label
    for label, class_id in destination_mapping.items()
}

label_to_id = {
    label: int(class_id)
    for label, class_id in destination_mapping.items()
}

labels = [
    id_to_label[index]
    for index in sorted(id_to_label)
]

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

dataset = SentraDataset(TEST_FILE)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_batch
)

# --------------------------------------------------
# LOAD CHECKPOINT AND MODEL
# --------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE,
    weights_only=True
)

model = SentraExtractor(
    vocab_size=VOCAB_SIZE,
    padding_idx=PAD_ID
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

actual_labels = []
predicted_labels = []
incorrect_records = []

record_index = 0

with torch.no_grad():

    for input_ids, lengths, targets in loader:

        outputs = model(input_ids, lengths)

        logits = outputs["destination"]

        predictions = torch.argmax(
            logits,
            dim=1
        )

        for batch_index in range(len(input_ids)):

            encoded = dataset.records[record_index]
            original = encoded["original_record"]

            actual = original["destination"]

            predicted = id_to_label[
                int(predictions[batch_index].item())
            ]

            actual_labels.append(actual)
            predicted_labels.append(predicted)

            if actual != predicted:

                incorrect_records.append({
                    "example_id": original["example_id"],
                    "group_id": original["group_id"],
                    "prompt": original["prompt"],
                    "actual": actual,
                    "predicted": predicted
                })

            record_index += 1

# --------------------------------------------------
# OVERALL ACCURACY
# --------------------------------------------------

accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)

print("\n===== DESTINATION EVALUATION: TEST1 =====")

print("Total records:", len(actual_labels))

print(
    f"Overall accuracy: {accuracy:.4%}"
)

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=labels,
        zero_division=0,
        digits=4
    )
)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

matrix = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=labels
)

print("\n===== CONFUSION MATRIX =====")

print("Rows = Actual, Columns = Predicted")

print("Labels:", labels)

for label, row in zip(labels, matrix):

    print(f"{label:25}", row.tolist())

# --------------------------------------------------
# ACTUAL VS PREDICTED DISTRIBUTION
# --------------------------------------------------

actual_counts = Counter(actual_labels)
predicted_counts = Counter(predicted_labels)

print("\n===== DESTINATION DISTRIBUTION =====")

for label in labels:

    print(
        f"{label:25} "
        f"Actual={actual_counts[label]:4} "
        f"Predicted={predicted_counts[label]:4}"
    )

# --------------------------------------------------
# INCORRECT PREDICTIONS
# --------------------------------------------------

print("\n===== INCORRECT PREDICTIONS =====")

print("Total errors:", len(incorrect_records))

for item in incorrect_records:

    print("\nExample:", item["example_id"])
    print("Group:", item["group_id"])
    print("Prompt:", item["prompt"])
    print("Actual:", item["actual"])
    print("Predicted:", item["predicted"])

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

output_file = ROOT / "logs" / "destination_test1_errors.jsonl"

with output_file.open("w", encoding="utf-8") as file:

    for item in incorrect_records:

        file.write(
            json.dumps(item, ensure_ascii=False) + "\n"
        )

print("\nError details saved to:")
print(output_file)

print("\n===== EVALUATION COMPLETED =====")
