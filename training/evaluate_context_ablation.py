

import sys
import json
import torch
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tokenizer"))

from sentra_extractor import SentraExtractor
from hybrid_bpe import load_hybrid_tokenizer, encode_hybrid

DEVICE = torch.device("cpu")

PAD_ID = 1988
VOCAB_SIZE = 1989
THRESHOLD = 0.5

TEST_FILE = (
    ROOT / "data" / "intent" / "encoded" / "test1_encoded.jsonl"
)

CHECKPOINT_FILE = ROOT / "checkpoints" / "sentra_best.pt"
MAPPING_FILE = ROOT / "configs" / "label_mappings.json"

WORD_MERGES = ROOT / "tokenizer" / "merges_word_v3.txt"
PHRASE_MERGES = ROOT / "tokenizer" / "merges_phrase_v4.txt"


# --------------------------------------------------
# LOAD LABEL MAPPINGS
# --------------------------------------------------

with MAPPING_FILE.open("r", encoding="utf-8") as file:
    mappings = json.load(file)

id_to_label = {
    field: {
        int(class_id): label
        for label, class_id in mapping.items()
    }
    for field, mapping in mappings.items()
}


# --------------------------------------------------
# LOAD TOKENIZER
# --------------------------------------------------

(
    word_merges,
    phrase_merges,
    vocabulary,
    tokenizer_vocab_size
) = load_hybrid_tokenizer(
    WORD_MERGES,
    PHRASE_MERGES
)

assert tokenizer_vocab_size == PAD_ID, (
    f"Tokenizer vocabulary mismatch: "
    f"{tokenizer_vocab_size} != {PAD_ID}"
)


def encode_prompt(context, prompt):

    text = (
        "[CONTEXT] "
        + json.dumps(context, ensure_ascii=False)
        + " [PROMPT] "
        + prompt
    )

    return encode_hybrid(
        text,
        word_merges,
        phrase_merges
    )


# --------------------------------------------------
# LOAD TRAINED MODEL
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
# DATASET
# --------------------------------------------------

class AblationDataset(Dataset):

    def __init__(self, records, remove_context):

        self.records = records
        self.remove_context = remove_context

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        encoded_record = self.records[index]

        # Original 14-field record is nested here.
        original = encoded_record["original_record"]

        context = (
            []
            if self.remove_context
            else original["conversation_context"]
        )

        token_ids = encode_prompt(
            context,
            original["prompt"]
        )

        return {
            "input_ids": torch.tensor(
                token_ids,
                dtype=torch.long
            )
        }


def ablation_collate(batch):

    input_ids = [
        item["input_ids"]
        for item in batch
    ]

    lengths = torch.tensor(
        [len(ids) for ids in input_ids],
        dtype=torch.long
    )

    padded = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=PAD_ID
    )

    return padded, lengths


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate(records, remove_context):

    dataset = AblationDataset(
        records,
        remove_context
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=ablation_collate
    )

    correct = 0
    total = 0
    wrong_examples = []

    processed = 0

    with torch.no_grad():

        for input_ids, lengths in loader:

            outputs = model(
                input_ids,
                lengths
            )

            for batch_index in range(len(input_ids)):

                encoded_record = records[processed]
                original = encoded_record["original_record"]

                processed += 1

                # Keep the same 113 multi-intent records
                # in both experiments.
                if len(original["intent_labels"]) <= 1:
                    continue

                actual = original["destination"]

                logits = outputs["destination"][batch_index]

                predicted_id = int(
                    torch.argmax(logits).item()
                )

                predicted = id_to_label["destination"][
                    predicted_id
                ]

                total += 1

                if predicted == actual:

                    correct += 1

                else:

                    wrong_examples.append({
                        "example_id": original["example_id"],
                        "group_id": original["group_id"],
                        "prompt": original["prompt"],
                        "actual": actual,
                        "predicted": predicted
                    })

    accuracy = correct / total if total else 0.0

    return accuracy, correct, total, wrong_examples


# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

with TEST_FILE.open("r", encoding="utf-8") as file:

    records = [
        json.loads(line)
        for line in file
    ]


multi_intent_count = sum(
    1
    for record in records
    if len(record["original_record"]["intent_labels"]) > 1
)

print("\n===== CONTEXT ABLATION EXPERIMENT =====")

print("Test1 records:", len(records))
print("Multi-intent records:", multi_intent_count)


# --------------------------------------------------
# EXPERIMENT A: ORIGINAL CONTEXT
# --------------------------------------------------

accuracy_a, correct_a, total_a, errors_a = evaluate(
    records,
    remove_context=False
)

print("\n--- EXPERIMENT A: ORIGINAL CONTEXT ---")

print(f"Correct: {correct_a}/{total_a}")
print(f"Destination accuracy: {accuracy_a:.4%}")


# --------------------------------------------------
# EXPERIMENT B: EMPTY CONTEXT
# --------------------------------------------------

accuracy_b, correct_b, total_b, errors_b = evaluate(
    records,
    remove_context=True
)

print("\n--- EXPERIMENT B: EMPTY CONTEXT ---")

print(f"Correct: {correct_b}/{total_b}")
print(f"Destination accuracy: {accuracy_b:.4%}")


# --------------------------------------------------
# COMPARISON
# --------------------------------------------------

print("\n--- COMPARISON ---")

difference = accuracy_a - accuracy_b

print(
    f"Accuracy difference: "
    f"{difference * 100:.2f} percentage points"
)


# --------------------------------------------------
# ERROR DETAILS
# --------------------------------------------------

print("\n--- ERRORS WITH ORIGINAL CONTEXT ---")

for item in errors_a:

    print(
        item["example_id"],
        "| Actual:", item["actual"],
        "| Predicted:", item["predicted"]
    )


print("\n--- ERRORS WITH EMPTY CONTEXT ---")

for item in errors_b:

    print(
        item["example_id"],
        "| Actual:", item["actual"],
        "| Predicted:", item["predicted"]
    )


print("\n===== EXPERIMENT COMPLETED =====")
