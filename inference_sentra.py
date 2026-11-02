
import sys
import json
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tokenizer"))

from sentra_extractor import SentraExtractor
from hybrid_bpe import load_hybrid_tokenizer, encode_hybrid


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DEVICE = torch.device("cpu")

PAD_ID = 1988
VOCAB_SIZE = 1989

CHECKPOINT_FILE = ROOT / "checkpoints" / "sentra_best.pt"

WORD_MERGES_FILE = ROOT / "tokenizer" / "merges_word_v3.txt"
PHRASE_MERGES_FILE = ROOT / "tokenizer" / "merges_phrase_v4.txt"

MAPPINGS_FILE = ROOT / "configs" / "label_mappings.json"

MULTI_FIELDS = [
    "intent_labels",
    "action",
    "data_type"
]

# --------------------------------------------------
# LOAD LABEL MAPPINGS
# --------------------------------------------------

with MAPPINGS_FILE.open("r", encoding="utf-8") as file:
    LABEL_MAPPINGS = json.load(file)

ID_TO_LABEL = {
    field: {
        int(class_id): label
        for label, class_id in mapping.items()
    }
    for field, mapping in LABEL_MAPPINGS.items()
}


# --------------------------------------------------
# LOAD TOKENIZER
# --------------------------------------------------

(
    WORD_MERGES,
    PHRASE_MERGES,
    VOCABULARY,
    TOKENIZER_VOCAB_SIZE
) = load_hybrid_tokenizer(
    WORD_MERGES_FILE,
    PHRASE_MERGES_FILE
)

if TOKENIZER_VOCAB_SIZE != PAD_ID:
    raise ValueError(
        f"Tokenizer vocabulary mismatch: "
        f"{TOKENIZER_VOCAB_SIZE} != {VOCAB_SIZE}"
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

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict(context, prompt):

    if not isinstance(context, list):
        raise TypeError("context must be a list of messages.")

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    context_text = json.dumps(
        context,
        ensure_ascii=False
    )

    model_input = (
        "[CONTEXT] "
        + context_text
        + " [PROMPT] "
        + prompt
    )

    input_ids = encode_hybrid(
        model_input,
        WORD_MERGES,
        PHRASE_MERGES
    )

    if not input_ids:
        raise ValueError("Tokenizer returned an empty sequence.")

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=DEVICE
    )

    lengths = torch.tensor(
        [len(input_ids)],
        dtype=torch.long
    )

    with torch.no_grad():
        outputs = model(input_tensor, lengths)

    predictions = {}

    for field, logits in outputs.items():

        if field in MULTI_FIELDS:

            probabilities = torch.sigmoid(logits)[0]

            predicted_ids = (
                probabilities >= 0.5
            ).nonzero(as_tuple=True)[0].tolist()

            predictions[field] = [
                ID_TO_LABEL[field][class_id]
                for class_id in predicted_ids
            ]

            predictions[field + "_confidence"] = [
                round(probabilities[class_id].item(), 4)
                for class_id in predicted_ids
            ]

        else:

            probabilities = torch.softmax(
                logits[0],
                dim=0
            )

            predicted_id = int(
                torch.argmax(probabilities).item()
            )

            predictions[field] = (
                ID_TO_LABEL[field][predicted_id]
            )

            predictions[field + "_confidence"] = round(
                probabilities[predicted_id].item(),
                4
            )

    return predictions

# --------------------------------------------------
# COMPLETE 14-FIELD RECORD ASSEMBLY
# --------------------------------------------------

import uuid


def build_record(context, prompt, user_role, session_id):

    # Validate inputs
    if not isinstance(context, list):
        raise TypeError("context must be a list of messages.")

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    if not isinstance(user_role, str) or not user_role.strip():
        raise ValueError("user_role must be a non-empty string.")

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string.")

    # Run the existing extraction model
    predictions = predict(context, prompt)

    # Assemble the agreed 14-field record
    record = {
        "example_id": str(uuid.uuid4()),
        "group_id": session_id,

        "scenario_family": predictions["scenario_family"],

        "conversation_context": context,
        "prompt": prompt,

        "intent_labels": predictions["intent_labels"],
        "action": predictions["action"],
        "target": predictions["target"],
        "data_type": predictions["data_type"],

        # Reserved for the decision engine
        "sensitivity": None,

        "destination": predictions["destination"],
        "authorization": predictions["authorization"],
        "context_dependency": (
            predictions["context_dependency"] == "True"
        ),

        # Supplied by the authenticated session
        "user_role": user_role
    }

    # Keep confidence scores separate from the 14-field record
    confidence = {
        key: value
        for key, value in predictions.items()
        if key.endswith("_confidence")
    }

    return {
        "record": record,
        "confidence": confidence
    }



# --------------------------------------------------
# COMMAND-LINE TEST
# --------------------------------------------------


# --------------------------------------------------
# COMMAND-LINE TEST
# --------------------------------------------------


# --------------------------------------------------
# COMMAND-LINE TEST
# --------------------------------------------------

if __name__ == "__main__":

    print("===== SENTRA RECORD ASSEMBLY TEST =====")

    context = []

    prompt = input("\nEnter a prompt: ").strip()

    # Temporary test fixture.
    # In the integrated application, this comes from
    # the authenticated session, never from user input.
    authenticated_user_role = "software engineer"

    # One session ID reused across requests in this session.
    session_id = "SESSION_001"

    result = build_record(
        context=context,
        prompt=prompt,
        user_role=authenticated_user_role,
        session_id=session_id
    )

    print("\n===== COMPLETE 14-FIELD RECORD =====")

    print(json.dumps(
        result["record"],
        indent=4,
        ensure_ascii=False
    ))

    print("\n===== EXTRACTION CONFIDENCE =====")

    print(json.dumps(
        result["confidence"],
        indent=4,
        ensure_ascii=False
    ))

    print("\nTotal record fields:", len(result["record"]))
