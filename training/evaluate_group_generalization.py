
import sys
import json
import torch
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "training"))

from sentra_extractor import SentraExtractor
from train_sentra import SentraDataset, collate_batch

DEVICE = torch.device("cpu")
PAD_ID = 1988
VOCAB_SIZE = 1989
THRESHOLD = 0.5

TEST_FILE = ROOT / "data" / "intent" / "encoded" / "test1_encoded.jsonl"
CHECKPOINT_FILE = ROOT / "checkpoints" / "sentra_best.pt"
MAPPING_FILE = ROOT / "configs" / "label_mappings.json"

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

with MAPPING_FILE.open("r", encoding="utf-8") as file:
    mappings = json.load(file)

id_to_label = {
    field: {
        int(class_id): label
        for label, class_id in mapping.items()
    }
    for field, mapping in mappings.items()
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

groups = defaultdict(list)
field_correct = defaultdict(int)
field_total = defaultdict(int)

record_exact = 0
multi_record_total = 0
record_index = 0


def decode_prediction(field, logits):

    if field in MULTI_FIELDS:

        probabilities = torch.sigmoid(logits)

        ids = (
            probabilities >= THRESHOLD
        ).nonzero(as_tuple=True)[0].tolist()

        return {
            id_to_label[field][class_id]
            for class_id in ids
        }

    predicted_id = int(torch.argmax(logits).item())

    return id_to_label[field][predicted_id]


with torch.no_grad():

    for input_ids, lengths, targets in loader:

        outputs = model(input_ids, lengths)

        for batch_index in range(len(input_ids)):

            encoded = dataset.records[record_index]
            original = encoded["original_record"]

            record_index += 1

            # Only inspect multi-intent records.
            if len(original["intent_labels"]) <= 1:
                continue

            multi_record_total += 1

            actual_fields = {}
            predicted_fields = {}

            for field in MULTI_FIELDS:

                actual_ids = set(
                    encoded["labels"][field]
                )

                actual = {
                    id_to_label[field][class_id]
                    for class_id in actual_ids
                }

                predicted = decode_prediction(
                    field,
                    outputs[field][batch_index]
                )

                actual_fields[field] = actual
                predicted_fields[field] = predicted

            for field in SINGLE_FIELDS:

                actual_id = encoded["labels"][field]

                actual_fields[field] = (
                    id_to_label[field][actual_id]
                )

                predicted_fields[field] = decode_prediction(
                    field,
                    outputs[field][batch_index]
                )

            record_is_correct = True

            for field in MULTI_FIELDS + SINGLE_FIELDS:

                field_total[field] += 1

                if actual_fields[field] == predicted_fields[field]:
                    field_correct[field] += 1
                else:
                    record_is_correct = False

            if record_is_correct:
                record_exact += 1

            groups[original["group_id"]].append({
                "example_id": original["example_id"],
                "prompt": original["prompt"],
                "actual": actual_fields,
                "predicted": predicted_fields,
                "correct": record_is_correct
            })


# --------------------------------------------------
# REPORT
# --------------------------------------------------

print("\n===== GROUP GENERALIZATION EVALUATION =====")

print("Multi-intent records:", multi_record_total)
print("Unique groups:", len(groups))

print("\n===== FIELD-LEVEL ACCURACY =====")

for field in MULTI_FIELDS + SINGLE_FIELDS:

    accuracy = field_correct[field] / field_total[field]

    print(
        f"{field:25} "
        f"{field_correct[field]}/{field_total[field]} "
        f"({accuracy:.4%})"
    )

print("\n===== COMPLETE RECORD ACCURACY =====")

print(
    "Records correct across all 8 fields:",
    f"{record_exact}/{multi_record_total}"
)

print(
    "Record exact match:",
    f"{record_exact / multi_record_total:.4%}"
)

group_exact = 0

for group_id, records in groups.items():

    if all(record["correct"] for record in records):
        group_exact += 1

print("\n===== GROUP-LEVEL ACCURACY =====")

print("Groups with every record correct:", group_exact)
print("Total groups:", len(groups))
print("Group exact match:", f"{group_exact / len(groups):.4%}")


# --------------------------------------------------
# PRINT INCORRECT GROUPS
# --------------------------------------------------

print("\n===== GROUPS CONTAINING ERRORS =====")

for group_id, records in groups.items():

    if all(record["correct"] for record in records):
        continue

    print("\nGROUP:", group_id)

    for record in records:

        if record["correct"]:
            continue

        print("\nExample:", record["example_id"])
        print("Prompt:", record["prompt"])

        for field in MULTI_FIELDS + SINGLE_FIELDS:

            actual = record["actual"][field]
            predicted = record["predicted"][field]

            if actual != predicted:

                print(
                    f"{field}: "
                    f"Actual={actual} | "
                    f"Predicted={predicted}"
                )

print("\nEvaluation completed.")
