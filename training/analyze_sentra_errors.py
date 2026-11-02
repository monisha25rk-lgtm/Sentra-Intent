
import sys
import json
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "training"))

from sentra_extractor import SentraExtractor
from train_sentra import SentraDataset, collate_batch, BATCH_SIZE

CHECKPOINT = ROOT / "checkpoints" / "sentra_best.pt"
TEST_FILE = ROOT / "data" / "intent" / "encoded" / "test1_encoded.jsonl"
MAPPING_FILE = ROOT / "configs" / "label_mappings.json"
OUTPUT_FILE = ROOT / "logs" / "test1_error_analysis.jsonl"

PAD_ID = 1988
VOCAB_SIZE = 1989
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


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_mappings():
    with open(MAPPING_FILE, "r", encoding="utf-8") as file:
        mappings = json.load(file)

    return {
        field: {int(index): name for name, index in mapping.items()}
        for field, mapping in mappings.items()
    }


def decode_labels(ids, inverse_mapping):
    return [
        inverse_mapping[int(label_id)]
        for label_id in sorted(ids)
    ]


def main():
    print("=" * 65)
    print("SENTRA ERROR ANALYSIS")
    print("=" * 65)

    records = load_jsonl(TEST_FILE)
    mappings = load_mappings()

    dataset = SentraDataset(TEST_FILE)

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
    print("Test records:", len(records))

    errors = []
    error_counts = Counter()
    record_index = 0

    with torch.no_grad():
        for input_ids, lengths, targets in loader:
            outputs = model(input_ids.to(DEVICE), lengths)

            batch_size = input_ids.size(0)

            for batch_index in range(batch_size):
                original = records[record_index]["original_record"]
                incorrect_fields = {}

                for field in MULTI_LABEL_FIELDS:
                    probabilities = torch.sigmoid(
                        outputs[field][batch_index]
                    )

                    predicted_ids = set(
                        torch.where(probabilities >= 0.5)[0]
                        .cpu().tolist()
                    )

                    actual_ids = set(
                        torch.where(
                            targets[field][batch_index] == 1
                        )[0].cpu().tolist()
                    )

                    if predicted_ids != actual_ids:
                        incorrect_fields[field] = {
                            "actual": decode_labels(
                                actual_ids, mappings[field]
                            ),
                            "predicted": decode_labels(
                                predicted_ids, mappings[field]
                            )
                        }

                for field in SINGLE_LABEL_FIELDS:
                    predicted_id = outputs[field][batch_index].argmax().item()
                    actual_id = targets[field][batch_index].item()

                    if predicted_id != actual_id:
                        incorrect_fields[field] = {
                            "actual": mappings[field][actual_id],
                            "predicted": mappings[field][predicted_id]
                        }

                if incorrect_fields:
                    error_counts.update(incorrect_fields.keys())

                    errors.append({
                        "example_id": original["example_id"],
                        "group_id": original["group_id"],
                        "prompt": original["prompt"],
                        "conversation_context": original["conversation_context"],
                        "incorrect_fields": incorrect_fields
                    })

                record_index += 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for error in errors:
            file.write(
                json.dumps(error, ensure_ascii=False) + "\n"
            )

    print("\n" + "=" * 65)
    print("ERROR SUMMARY")
    print("=" * 65)

    print("Total incorrect records:", len(errors))
    print("Total test records:", len(records))
    print(
        "Records with no prediction errors:",
        len(records) - len(errors)
    )

    print("\nIncorrect predictions by field:")

    for field in SINGLE_LABEL_FIELDS + MULTI_LABEL_FIELDS:
        print(f"{field:<25} {error_counts[field]}")

    print("\n" + "=" * 65)
    print("FIRST 10 INCORRECT EXAMPLES")
    print("=" * 65)

    for error in errors[:10]:
        print("\nExample ID:", error["example_id"])
        print("Group ID:", error["group_id"])
        print("Prompt:", error["prompt"])

        for field, values in error["incorrect_fields"].items():
            print(f"\n{field}:")
            print("  Actual:   ", values["actual"])
            print("  Predicted:", values["predicted"])

    print("\nFull error report saved to:")
    print(OUTPUT_FILE)

    print("\nERROR ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
