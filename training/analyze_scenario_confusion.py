
import sys
import json
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "training"))

from sentra_extractor import SentraExtractor
from train_sentra import SentraDataset, collate_batch, BATCH_SIZE

CHECKPOINT = ROOT / "checkpoints" / "sentra_best.pt"
TEST_FILE = ROOT / "data" / "intent" / "encoded" / "test1_encoded.jsonl"
MAPPING_FILE = ROOT / "configs" / "label_mappings.json"

PAD_ID = 1988
VOCAB_SIZE = 1989
DEVICE = torch.device("cpu")


def main():
    print("=" * 70)
    print("SENTRA SCENARIO FAMILY CONFUSION ANALYSIS")
    print("=" * 70)

    with open(MAPPING_FILE, "r", encoding="utf-8") as file:
        mappings = json.load(file)

    scenario_mapping = mappings["scenario_family"]

    inverse_mapping = {
        int(index): name
        for name, index in scenario_mapping.items()
    }

    labels = sorted(inverse_mapping.keys())
    class_names = [inverse_mapping[i] for i in labels]

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

    actual_all = []
    predicted_all = []
    incorrect_examples = []

    records = []

    with open(TEST_FILE, "r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]

    index = 0

    with torch.no_grad():
        for input_ids, lengths, targets in loader:
            outputs = model(input_ids.to(DEVICE), lengths)

            predictions = outputs["scenario_family"].argmax(dim=1)

            actual = targets["scenario_family"].cpu().numpy()
            predicted = predictions.cpu().numpy()

            actual_all.extend(actual.tolist())
            predicted_all.extend(predicted.tolist())

            for true_id, predicted_id in zip(actual, predicted):
                if true_id != predicted_id:
                    record = records[index]["original_record"]

                    incorrect_examples.append({
                        "example_id": record["example_id"],
                        "group_id": record["group_id"],
                        "prompt": record["prompt"],
                        "actual": inverse_mapping[int(true_id)],
                        "predicted": inverse_mapping[int(predicted_id)]
                    })

                index += 1

    actual_all = np.array(actual_all)
    predicted_all = np.array(predicted_all)

    matrix = confusion_matrix(
        actual_all,
        predicted_all,
        labels=labels
    )

    print("\n===== CONFUSION MATRIX =====")
    print("Rows = Actual | Columns = Predicted\n")

    print(f"{'ACTUAL / PREDICTED':45}", end="")
    for name in class_names:
        print(f"{name[:12]:>15}", end="")
    print()

    for i, row in enumerate(matrix):
        print(f"{class_names[i][:44]:45}", end="")
        for value in row:
            print(f"{value:>15}", end="")
        print()

    print("\n===== PER-CLASS METRICS =====")
    print(
        classification_report(
            actual_all,
            predicted_all,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            digits=4
        )
    )

    print("\n===== MOST COMMON CONFUSIONS =====")

    confusion_counts = Counter()

    for example in incorrect_examples:
        pair = (example["actual"], example["predicted"])
        confusion_counts[pair] += 1

    for (actual, predicted), count in confusion_counts.most_common():
        print(f"{actual} -> {predicted}: {count}")

    print("\n===== INCORRECT EXAMPLES =====")

    for example in incorrect_examples:
        print("\nExample:", example["example_id"])
        print("Group:", example["group_id"])
        print("Prompt:", example["prompt"])
        print("Actual:", example["actual"])
        print("Predicted:", example["predicted"])

    print("\nTotal scenario-family errors:", len(incorrect_examples))
    print("Total records:", len(records))
    print("\nANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
