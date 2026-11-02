
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "training"))

from sentra_extractor import SentraExtractor
from sentra_losses import SentraMultiTaskLoss
from train_sentra import (
    SentraDataset,
    collate_batch,
    TRAIN_FILE,
    VALIDATION_FILE,
    BATCH_SIZE,
    DEVICE
)


def main():

    print("===== SENTRA TRAINING PIPELINE TEST =====")

    # 1. Load datasets.
    train_dataset = SentraDataset(TRAIN_FILE)
    validation_dataset = SentraDataset(VALIDATION_FILE)

    print("\nDataset loading: PASS")
    print("Training records:", len(train_dataset))
    print("Validation records:", len(validation_dataset))

    # 2. Build a small batch.
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0
    )

    input_ids, lengths, targets = next(iter(loader))

    print("\n===== BATCH CHECK =====")
    print("Input shape:", tuple(input_ids.shape))
    print("Lengths shape:", tuple(lengths.shape))

    for field, target in targets.items():
        print(f"{field:25} {tuple(target.shape)}")

    assert input_ids.ndim == 2
    assert len(lengths) == input_ids.size(0)
    assert input_ids.max().item() < 1989

    # 3. Initialize model, loss and optimizer.
    model = SentraExtractor().to(DEVICE)
    criterion = SentraMultiTaskLoss().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # 4. Forward pass.
    model.train()

    outputs = model(
        input_ids.to(DEVICE),
        lengths
    )

    # 5. Calculate all eight losses.
    targets = {
        field: value.to(DEVICE)
        for field, value in targets.items()
    }

    total_loss, individual_losses = criterion(
        outputs,
        targets
    )

    assert torch.isfinite(total_loss)

    print("\n===== LOSS CHECK =====")

    for field, loss in individual_losses.items():
        print(f"{field:25} {loss.item():.6f}")

    print("Total loss:", total_loss.item())

    # 6. Backpropagation.
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()

    # 7. Optimizer update.
    optimizer.step()

    print("\nForward pass: PASS")
    print("Loss calculation: PASS")
    print("Backpropagation: PASS")
    print("Optimizer update: PASS")

    print("\nALL TRAINING PIPELINE TESTS PASSED.")


if __name__ == "__main__":
    main()
