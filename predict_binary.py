"""
predict_binary.py — Run inference with a trained binary cognitive distortion model

Loads a checkpoint produced by Train_And_Evaluate_Binary.py (e.g.
results/results_<run_id>_best/<run_id>_seed_<seed>/) and predicts Distortion vs.
No Distortion for new, unlabeled text. The checkpoint is self-describing — label
names come from the model's own config.json (id2label), cross-checked against
label_classes.json in the same folder if present — so no separate label list needs
to be passed in.

Example usage:

    python predict_binary.py \
        --model_path results/results_gpt4_0.5_rlp_roberta_best/gpt4_0.5_rlp_roberta_seed_42 \
        --input_file new_data.csv \
        --text_column "Patient Question" \
        --output_file predictions.csv

Optional:
    --batch_size 16     inference batch size (default 16)
    --max_length 512    must match training (default 512, same as training script)
"""

import argparse
import json
import os

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser(description="Run binary cognitive distortion inference on new data")
parser.add_argument('--model_path', type=str, required=True,
                     help='Path to a trained checkpoint folder (contains config.json, pytorch_model.bin, tokenizer files)')
parser.add_argument('--input_file', type=str, required=True, help='CSV file containing the text to predict on')
parser.add_argument('--text_column', type=str, default='Patient Question', help='Column name containing the input text (default: "Patient Question")')
parser.add_argument('--output_file', type=str, default='predictions.csv', help='Where to write predictions (default: predictions.csv)')
parser.add_argument('--batch_size', type=int, default=16, help='Inference batch size (default: 16)')
parser.add_argument('--max_length', type=int, default=512, help='Max sequence length — must match training (default: 512)')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Load model + tokenizer
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
model.to(device)
model.eval()

# id2label keys come back as strings after a JSON round-trip; sort numerically
# so index i of the model's output vector maps to id2label[i].
id2label = {int(k): v for k, v in model.config.id2label.items()}
label_names = [id2label[i] for i in range(len(id2label))]

if len(label_names) != 2:
    raise ValueError(f"Expected a 2-class binary checkpoint, but {args.model_path} has "
                      f"{len(label_names)} classes: {label_names}. Is this the right model_path?")

# Cross-check against label_classes.json saved alongside the checkpoint, if present —
# catches a checkpoint whose config.json was hand-edited inconsistently with training.
label_classes_path = os.path.join(args.model_path, "label_classes.json")
if os.path.exists(label_classes_path):
    with open(label_classes_path) as f:
        saved_classes = json.load(f)
    if saved_classes != label_names:
        raise ValueError(
            f"Mismatch between model config id2label {label_names} and "
            f"label_classes.json {saved_classes} in {args.model_path} — "
            "this checkpoint's config may have been edited inconsistently."
        )

print(f"Loaded model with {len(label_names)} labels: {label_names}")

# ---------------------------------------------------------------------------
# Load input data
# ---------------------------------------------------------------------------
data = pd.read_csv(args.input_file)
if args.text_column not in data.columns:
    raise ValueError(f"--text_column '{args.text_column}' not found in {args.input_file}. "
                      f"Available columns: {list(data.columns)}")

texts = data[args.text_column].fillna("").astype(str).tolist()

# ---------------------------------------------------------------------------
# Predict in batches
# ---------------------------------------------------------------------------
all_predicted_labels = []
all_confidences = []
all_probabilities = []

with torch.no_grad():
    for start in range(0, len(texts), args.batch_size):
        batch_texts = texts[start:start + args.batch_size]
        encodings = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt"
        ).to(device)

        logits = model(**encodings).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for row_probs in probs:
            pred_idx = int(row_probs.argmax())
            all_predicted_labels.append(label_names[pred_idx])
            all_confidences.append(round(float(row_probs[pred_idx]), 4))
            all_probabilities.append(json.dumps(
                {label_names[i]: round(float(p), 4) for i, p in enumerate(row_probs)}
            ))

        print(f"Predicted {min(start + args.batch_size, len(texts))}/{len(texts)} rows")

data['predicted_label'] = all_predicted_labels
data['confidence'] = all_confidences
data['label_probabilities'] = all_probabilities
data.to_csv(args.output_file, index=False)
print(f"\nSaved predictions to {args.output_file}")
