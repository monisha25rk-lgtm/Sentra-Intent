
import sys
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "training"))

from sentra_extractor import SentraExtractor
from sentra_losses import SentraMultiTaskLoss


# ==========================================
# CONFIGURATION
# ==========================================

PAD_ID = 1988

BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 30
PATIENCE = 5

SEED = 42

DEVICE = torch.device("cpu")

TRAIN_FILE = (
    ROOT / "data" / "intent" / "encoded"
    / "train_grouped_encoded.jsonl"
)

VALIDATION_FILE = (
    ROOT / "data" / "intent" / "encoded"
    / "validation_encoded.jsonl"
)

CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"

CHECKPOINT_FILE = CHECKPOINT_DIR / "sentra_best.pt"
HISTORY_FILE = LOG_DIR / "training_history.jsonl"


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

LABEL_COUNTS = {
    "intent_labels": 9,
    "action": 9,
    "data_type": 14
}


# ==========================================
# REPRODUCIBILITY
# ==========================================

def set_seed():

    random.seed(SEED)
    torch.manual_seed(SEED)


# ==========================================
# DATASET
# ==========================================

class SentraDataset(Dataset):

    def __init__(self, file_path):

        self.records = []

        with file_path.open("r", encoding="utf-8") as file:

            for line in file:
                self.records.append(json.loads(line))

        if not self.records:
            raise ValueError(
                f"Dataset is empty: {file_path}"
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        return {
            "input_ids": torch.tensor(
                record["input_ids"],
                dtype=torch.long
            ),
            "labels": record["labels"]
        }


# ==========================================
# BATCH COLLATION
# ==========================================

def collate_batch(batch):

    input_sequences = [
        item["input_ids"]
        for item in batch
    ]

    lengths = torch.tensor(
        [len(sequence) for sequence in input_sequences],
        dtype=torch.long
    )

    input_ids = pad_sequence(
        input_sequences,
        batch_first=True,
        padding_value=PAD_ID
    )

    targets = {}

    # Multi-label fields become multi-hot vectors.
    for field in MULTI_FIELDS:

        target_tensor = torch.zeros(
            len(batch),
            LABEL_COUNTS[field],
            dtype=torch.float32
        )

        for index, item in enumerate(batch):

            for label_id in item["labels"][field]:
                target_tensor[index, label_id] = 1.0

        targets[field] = target_tensor

    # Single-label fields remain integer class IDs.
    for field in SINGLE_FIELDS:

        targets[field] = torch.tensor(
            [
                item["labels"][field]
                for item in batch
            ],
            dtype=torch.long
        )

    return input_ids, lengths, targets


# ==========================================
# ONE TRAINING OR VALIDATION EPOCH
# ==========================================

def run_epoch(model, loader, criterion, optimizer=None):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss_sum = 0.0
    individual_loss_sums = {}

    total_records = 0

    for input_ids, lengths, targets in loader:

        input_ids = input_ids.to(DEVICE)

        targets = {
            field: value.to(DEVICE)
            for field, value in targets.items()
        }

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            outputs = model(input_ids, lengths)

            total_loss, individual_losses = criterion(
                outputs,
                targets
            )

            if not torch.isfinite(total_loss):
                raise ValueError(
                    "Non-finite loss detected."
                )

            if training:

                total_loss.backward()

                clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

        batch_size = input_ids.size(0)

        total_loss_sum += (
            total_loss.item() * batch_size
        )

        for field, loss in individual_losses.items():

            individual_loss_sums[field] = (
                individual_loss_sums.get(field, 0.0)
                + loss.item() * batch_size
            )

        total_records += batch_size

    average_total_loss = total_loss_sum / total_records

    average_individual_losses = {
        field: value / total_records
        for field, value in individual_loss_sums.items()
    }

    return average_total_loss, average_individual_losses


# ==========================================
# MAIN TRAINING FUNCTION
# ==========================================

def main():

    set_seed()

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("===== SENTRA TRAINING ENGINE =====")
    print("Device:", DEVICE)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Maximum epochs:", MAX_EPOCHS)

    # Load datasets.
    print("\nLoading datasets...")

    train_dataset = SentraDataset(TRAIN_FILE)
    validation_dataset = SentraDataset(VALIDATION_FILE)

    print("Training records:", len(train_dataset))
    print("Validation records:", len(validation_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0
    )

    # Initialize model and losses.
    print("\nInitializing model...")

    model = SentraExtractor().to(DEVICE)

    criterion = SentraMultiTaskLoss().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(
        "Model parameters:",
        sum(p.numel() for p in model.parameters())
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    # Start a fresh history file for this run.
    with HISTORY_FILE.open("w", encoding="utf-8") as history:

        for epoch in range(1, MAX_EPOCHS + 1):

            print(f"\n{'=' * 45}")
            print(f"EPOCH {epoch}/{MAX_EPOCHS}")
            print(f"{'=' * 45}")

            # Training.
            train_loss, train_details = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )

            # Validation.
            validation_loss, validation_details = run_epoch(
                model,
                validation_loader,
                criterion
            )

            print(f"\nTraining loss:   {train_loss:.6f}")
            print(f"Validation loss: {validation_loss:.6f}")

            print("\nIndividual validation losses:")

            for field, loss in validation_details.items():
                print(f"{field:25} {loss:.6f}")

            improved = (
                validation_loss < best_validation_loss
            )

            if improved:

                best_validation_loss = validation_loss
                epochs_without_improvement = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "validation_loss": validation_loss,
                        "vocab_size": 1989,
                        "padding_id": PAD_ID,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE
                    },
                    CHECKPOINT_FILE
                )

                print("\nBest model checkpoint saved.")

            else:

                epochs_without_improvement += 1

                print(
                    "\nNo improvement:",
                    epochs_without_improvement,
                    "/",
                    PATIENCE
                )

            epoch_result = {
                "epoch": epoch,
                "training_loss": train_loss,
                "validation_loss": validation_loss,
                "training_individual_losses": train_details,
                "validation_individual_losses": validation_details
            }

            history.write(
                json.dumps(epoch_result) + "\n"
            )

            history.flush()

            if epochs_without_improvement >= PATIENCE:

                print("\nEarly stopping triggered.")
                break

    print("\n===== TRAINING COMPLETED =====")
    print("Best validation loss:", best_validation_loss)
    print("Checkpoint:", CHECKPOINT_FILE)
    print("Training history:", HISTORY_FILE)


if __name__ == "__main__":
    main()
