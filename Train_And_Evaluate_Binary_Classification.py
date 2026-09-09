"""
Train_And_Evaluate_Binary_Classification.py — Filter, Split, Train, and Evaluate (Binary)

Takes a raw Final_Labels configuration CSV (Id_Number, Patient Question, RLP_Label,
MLP_Label, golden_labels), filters out ambiguous-label rows, derives a binary label
(Distortion vs. No Distortion) per row from the chosen label column, fine-tunes a
MentalBERT/MentalRoBERTa binary classifier across multiple seeds, and saves
self-describing checkpoints (config.json carries id2label/label2id) ready for
inference.

Binary label derivation: a row's label string is parsed into a list and the negative
label is dropped (same helper used by the multilabel/multiclass scripts) — if
anything remains, the row is labeled 1 (Distortion), otherwise 0 (No Distortion).


SEEDS — two independent concepts, don't confuse them:
  --seeds             List of model-training seeds (default: the paper's 5 seeds).
                       The SAME train/dev/test split is reused for every seed in this
                       list; only model init / dropout / batch order change. We train
                       once per seed and report mean +/- std, because a single run's
                       score is noisy on its own.
  --random_split_seed Single seed controlling how the DATA is partitioned. Only used
                       when --split_method random. Irrelevant under fixed_ids.

SPLIT METHOD:
  fixed_ids (default) Reproduces the paper's exact train/dev/test assignment via the
                       ID lists in splits/. Required for results comparable to the
                       paper. Row COUNT per split still differs per config, because
                       each config's own ambiguous-label filtering removes a
                       different subset of rows before this split is applied.
  random               Fresh split on this config's filtered data. NOT comparable to
                       the paper's reported numbers.

Example usage:

  Full paper-matching run (5 seeds, fixed split):
    python Train_And_Evaluate_Binary_Classification.py \
        --config_file Final_Labels/complete_data_gpt4_0.5t.csv \
        --model_type roberta --label_type rlp \
        --split_method fixed_ids \
        --train_ids splits/train_ids.txt --dev_ids splits/dev_ids.txt --test_ids splits/test_ids.txt

  Quick single-seed smoke test, fresh random split:
    python Train_And_Evaluate_Binary_Classification.py \
        --config_file Final_Labels/complete_data_gpt4_0.5t.csv \
        --model_type bert --label_type golden \
        --split_method random --seeds 42 --num_train_epochs 3

Outputs (all namespaced by run_id = <config_file basename>_<label_type>_<model_type>):
  results/results_<run_id>_best/<run_id>_seed_<seed>/   trained model + tokenizer + label_classes.json
  results/<run_id>_results.txt / .json                  per-seed classification reports + aggregated mean/std
  predictions_output/<run_id>_seed_<seed>_{val,test}_predictions.json
  predictions_output/<run_id>_seed_<seed>_confusion_matrix.png
  predictions_output/<run_id>_{validation,test}_metrics.csv
"""

import argparse
import inspect
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Filter, split, train, and evaluate a binary cognitive distortion classifier"
)
parser.add_argument('--config_file', type=str, required=True,
                     help='Path to the raw Final_Labels configuration CSV')
parser.add_argument('--model_type', type=str, required=True, choices=['bert', 'roberta'],
                     help='Model type')
parser.add_argument('--label_type', type=str, required=True, choices=['rlp', 'mlp', 'golden'],
                     help='Which label column to derive the binary label from')
parser.add_argument('--negative_label', type=str, default='No Distortion',
                     help='Label meaning "no distortion" (default: "No Distortion")')
parser.add_argument('--ambiguous_labels', type=str, nargs='+',
                     default=['Not sure which distortion', 'Not sure if distortion or not', 'Others'],
                     help='Labels that mark a row as ambiguous; rows with any of these in '
                          'RLP_Label or MLP_Label are dropped entirely')

parser.add_argument('--split_method', type=str, default='fixed_ids', choices=['fixed_ids', 'random'],
                     help="fixed_ids (default): reproduce the paper's exact train/dev/test "
                          "assignment using the ID lists in splits/ — required for results "
                          "comparable to the paper. random: fresh split on this config's "
                          "filtered data — NOT comparable to the paper's reported numbers.")
parser.add_argument('--train_ids', type=str, help='Text file, one Id_Number per line (required for --split_method fixed_ids)')
parser.add_argument('--dev_ids', type=str, help='Text file, one Id_Number per line (required for --split_method fixed_ids)')
parser.add_argument('--test_ids', type=str, help='Text file, one Id_Number per line (required for --split_method fixed_ids)')
parser.add_argument('--train_size', type=float, default=0.70, help='Train fraction for --split_method random (default: 0.70)')
parser.add_argument('--dev_size', type=float, default=0.15, help='Dev fraction for --split_method random (default: 0.15)')
parser.add_argument('--test_size', type=float, default=0.15, help='Test fraction for --split_method random (default: 0.15)')
parser.add_argument('--random_split_seed', type=int, default=42,
                     help='Seed controlling the data split itself; only used with --split_method random (default: 42)')

parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024],
                     help='Model-training seeds (default: the paper\'s 5 seeds). Pass one value, '
                          'e.g. --seeds 42, for a quick single-run test. Same split used for every seed.')

parser.add_argument('--num_train_epochs', type=int, default=100, help='Max training epochs (default: 100, as used in the paper)')
parser.add_argument('--train_batch_size', type=int, default=16, help='Per-device train batch size (default: 16)')
parser.add_argument('--eval_batch_size', type=int, default=16, help='Per-device eval batch size (default: 16)')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate (default: 5e-5)')
parser.add_argument('--eval_strategy', type=str, default='epoch', choices=['epoch', 'steps'],
                     help='Evaluation/save strategy (default: epoch)')
parser.add_argument('--metric_for_best_model', type=str, default='f1',
                     choices=['f1', 'f1_macro', 'f1_weighted', 'accuracy'],
                     help='Metric used to select the best checkpoint (default: f1 — binary f1 on '
                          'the "Distortion" positive class, as used in the paper)')
parser.add_argument('--early_stopping_patience', type=int, default=3,
                     help='Early stopping patience in evaluation rounds (default: 3)')

args = parser.parse_args()

MODEL_PATH = {
    'bert': 'mental/mental-bert-base-uncased',
    'roberta': 'mental/mental-roberta-base'
}
AUTH_TOKEN = "your authentication key"

LABEL_COLUMN_MAP = {
    'rlp': 'RLP_Label',
    'mlp': 'MLP_Label',
    'golden': 'golden_labels'
}
SEEDS = args.seeds

# Binary labels are fixed by construction (2 classes, derived from presence/
# absence of any non-negative label) rather than derived from the data, so
# id2label/label2id are deterministic regardless of config/label_type.
id2label = {0: args.negative_label, 1: 'Distortion'}
label2id = {args.negative_label: 0, 'Distortion': 1}
DISPLAY_LABELS = [args.negative_label, 'Distortion']

# run_id namespaces every output path so runs on different config files never
# overwrite each other's checkpoints/results.
config_name = os.path.splitext(os.path.basename(args.config_file))[0]
run_id = f"{config_name}_{args.label_type}_{args.model_type}"

RESULTS_DIR = "./results"
PREDICTIONS_DIR = "./predictions_output"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_label(label):
    return label.strip().lower()


def parse_labels(label_str):
    if not isinstance(label_str, str):
        return []
    return [lab.strip() for lab in label_str.split(',') if lab.strip()]


def drop_negative(labels_list, negative_label):
    neg_norm = normalize_label(negative_label)
    return [lab for lab in labels_list if normalize_label(lab) != neg_norm]


def load_ids(id_file):
    with open(id_file) as f:
        return {line.strip() for line in f if line.strip()}


class PreTokenizedMentalHealthDataset(Dataset):
    def __init__(self, encodings, labels, ids):
        self.encodings = {k: torch.tensor(v) for k, v in encodings.items()}
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.ids = torch.tensor(ids, dtype=torch.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = self.labels[idx]
        item['id'] = self.ids[idx]
        return item


def tokenize_dataset(dataset, tokenizer):
    return tokenizer(
        dataset["Patient Question"].tolist(),
        padding="max_length", truncation=True, max_length=512, return_tensors="pt"
    )


def compute_metrics(pred):
    labels = pred.label_ids
    predictions = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average='binary', pos_label=1),
        "precision": precision_score(labels, predictions, average='binary', pos_label=1),
        "recall": recall_score(labels, predictions, average='binary', pos_label=1),
        "f1_macro": f1_score(labels, predictions, average='macro'),
        "f1_weighted": f1_score(labels, predictions, average='weighted'),
        "precision_macro": precision_score(labels, predictions, average='macro'),
        "precision_weighted": precision_score(labels, predictions, average='weighted'),
        "recall_macro": recall_score(labels, predictions, average='macro'),
        "recall_weighted": recall_score(labels, predictions, average='weighted'),
    }


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average='binary', pos_label=1),
        "precision": precision_score(y_true, y_pred, average='binary', pos_label=1),
        "recall": recall_score(y_true, y_pred, average='binary', pos_label=1),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
        "precision_macro": precision_score(y_true, y_pred, average='macro'),
        "precision_weighted": precision_score(y_true, y_pred, average='weighted'),
        "recall_macro": recall_score(y_true, y_pred, average='macro'),
        "recall_weighted": recall_score(y_true, y_pred, average='weighted'),
    }


def plot_confusion_matrices(y_true_val, y_pred_val, y_true_test, y_pred_test, run_id, seed):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Confusion Matrices — {run_id} (Seed {seed})")

    cm_val = confusion_matrix(y_true_val, y_pred_val, labels=[0, 1])
    ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=DISPLAY_LABELS).plot(ax=axes[0], cmap='Blues')
    axes[0].set_title("Validation Set")

    cm_test = confusion_matrix(y_true_test, y_pred_test, labels=[0, 1])
    ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=DISPLAY_LABELS).plot(ax=axes[1], cmap='Blues')
    axes[1].set_title("Test Set")

    plt.tight_layout()
    plt.savefig(f"{PREDICTIONS_DIR}/{run_id}_seed_{seed}_confusion_matrix.png")
    plt.close()


# ---------------------------------------------------------------------------
# Load + filter the raw configuration file
# ---------------------------------------------------------------------------
data = pd.read_csv(args.config_file)

# float -> int -> str avoids pandas upcasting Id_Number to "12345.0" when the
# column has any missing values, which would silently break every ID match below.
data['Id_Number'] = data['Id_Number'].astype(float).astype(int).astype(str)

ambiguous_norm = {normalize_label(l) for l in args.ambiguous_labels}


def row_is_clean(row):
    for col in [LABEL_COLUMN_MAP['rlp'], LABEL_COLUMN_MAP['mlp']]:
        if pd.isna(row[col]):
            continue
        row_labels = {normalize_label(l) for l in parse_labels(row[col])}
        if row_labels & ambiguous_norm:
            return False
    return True


before_count = len(data)
data = data[data.apply(row_is_clean, axis=1)].reset_index(drop=True)
print(f"[{run_id}] Filtered ambiguous-label rows: {before_count} -> {len(data)} rows")

# Binary label: parse the source column, drop the negative label, and check
# whether anything real remains. A NaN/unparseable source value can't be
# resolved to a binary label, so those rows are dropped (not silently
# defaulted to 0) — this is the case the original substring check would have
# crashed on.
source_column = LABEL_COLUMN_MAP[args.label_type]

before_count = len(data)
data = data[data[source_column].apply(lambda x: isinstance(x, str))].reset_index(drop=True)
if before_count != len(data):
    print(f"[{run_id}] WARNING: dropped {before_count - len(data)} rows with an empty/unparseable {source_column}")

data['binary_label'] = data[source_column].apply(
    lambda x: 1 if drop_negative(parse_labels(x), args.negative_label) else 0
)

# ---------------------------------------------------------------------------
# Split into train/dev/test
# ---------------------------------------------------------------------------
if args.split_method == 'fixed_ids':
    if not (args.train_ids and args.dev_ids and args.test_ids):
        parser.error("--train_ids, --dev_ids, and --test_ids are required when --split_method fixed_ids")
    train_ids = load_ids(args.train_ids)
    dev_ids = load_ids(args.dev_ids)
    test_ids = load_ids(args.test_ids)

    overlap = (train_ids & dev_ids) | (train_ids & test_ids) | (dev_ids & test_ids)
    if overlap:
        parser.error(f"train_ids/dev_ids/test_ids overlap on {len(overlap)} Id_Number(s) — check your split files")

    train_data = data[data['Id_Number'].isin(train_ids)].reset_index(drop=True)
    val_data = data[data['Id_Number'].isin(dev_ids)].reset_index(drop=True)
    test_data = data[data['Id_Number'].isin(test_ids)].reset_index(drop=True)

    for name, split_df in [('train', train_data), ('dev', val_data), ('test', test_data)]:
        if len(split_df) == 0:
            parser.error(f"{name} split has 0 rows after filtering — check that {name}_ids matches this config's Id_Numbers")
else:
    train_data, temp_data = train_test_split(
        data, train_size=args.train_size, random_state=args.random_split_seed
    )
    relative_dev_size = args.dev_size / (args.dev_size + args.test_size)
    val_data, test_data = train_test_split(
        temp_data, train_size=relative_dev_size, random_state=args.random_split_seed
    )
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

print(f"[{run_id}] Split sizes -> train: {len(train_data)}, dev: {len(val_data)}, test: {len(test_data)}")

y_true_val = val_data['binary_label'].values
y_true_test = test_data['binary_label'].values

# ---------------------------------------------------------------------------
# Tokenize
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[args.model_type], use_auth_token=AUTH_TOKEN)
train_encodings = tokenize_dataset(train_data, tokenizer)
val_encodings = tokenize_dataset(val_data, tokenizer)
test_encodings = tokenize_dataset(test_data, tokenizer)

train_dataset = PreTokenizedMentalHealthDataset(
    train_encodings, train_data['binary_label'].values, train_data['Id_Number'].astype(int).values
)
val_dataset = PreTokenizedMentalHealthDataset(
    val_encodings, val_data['binary_label'].values, val_data['Id_Number'].astype(int).values
)
test_dataset = PreTokenizedMentalHealthDataset(
    test_encodings, test_data['binary_label'].values, test_data['Id_Number'].astype(int).values
)

# transformers version compatibility: newer releases renamed evaluation_strategy -> eval_strategy
_ta_params = inspect.signature(TrainingArguments.__init__).parameters
_eval_strategy_key = 'eval_strategy' if 'eval_strategy' in _ta_params else 'evaluation_strategy'

# ---------------------------------------------------------------------------
# Train + evaluate across seeds
# ---------------------------------------------------------------------------
results_file = os.path.join(RESULTS_DIR, f"{run_id}_results.txt")
results_file_json = os.path.join(RESULTS_DIR, f"{run_id}_results.json")

json_output = {}
train_time_metrics = []
val_metrics_list = []
test_metrics_list = []

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH[args.model_type],
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        use_auth_token=AUTH_TOKEN
    )
    if torch.cuda.is_available():
        model.cuda()

    training_args = TrainingArguments(
        output_dir=f'{RESULTS_DIR}/results_{run_id}/{run_id}_seed_{seed}',
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        save_strategy=args.eval_strategy,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        report_to="none",
        seed=seed,
        **{_eval_strategy_key: args.eval_strategy}
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )

    print(f"---------------- {run_id} ----------------")
    print(f"------------------- Training for seed {seed} -------------------\n")
    trainer.train()

    print(f"------------------- Evaluating on validation set for seed {seed} -------------------")
    print(trainer.evaluate(), "\n")

    model_path_to_save = f"{RESULTS_DIR}/results_{run_id}_best/{run_id}_seed_{seed}"
    model.save_pretrained(model_path_to_save)
    tokenizer.save_pretrained(model_path_to_save)
    with open(os.path.join(model_path_to_save, "label_classes.json"), "w") as f:
        json.dump(DISPLAY_LABELS, f, indent=2)

    val_predictions = trainer.predict(val_dataset)
    y_pred_val = np.argmax(val_predictions.predictions, axis=1)

    with open(results_file, "a") as f:
        f.write(f"Seed {seed} - Classification Report:\n")
        f.write(classification_report(y_true_val, y_pred_val, target_names=DISPLAY_LABELS, labels=[0, 1], digits=4) + "\n")

    classification_report_dict = classification_report(
        y_true_val, y_pred_val, target_names=DISPLAY_LABELS, labels=[0, 1], digits=4, output_dict=True
    )
    json_output[f"seed_{seed}"] = {"classification_report": classification_report_dict}

    train_time_metrics.append({
        "accuracy": classification_report_dict['accuracy'],
        "f1_macro": classification_report_dict['macro avg']['f1-score'],
        "f1_weighted": classification_report_dict['weighted avg']['f1-score'],
        "precision_macro": classification_report_dict['macro avg']['precision'],
        "precision_weighted": classification_report_dict['weighted avg']['precision'],
        "recall_macro": classification_report_dict['macro avg']['recall'],
        "recall_weighted": classification_report_dict['weighted avg']['recall'],
        "f1": classification_report_dict['Distortion']['f1-score'],
        "precision": classification_report_dict['Distortion']['precision'],
        "recall": classification_report_dict['Distortion']['recall'],
    })

    test_predictions = trainer.predict(test_dataset)
    y_pred_test = np.argmax(test_predictions.predictions, axis=1)

    json.dump(y_pred_val.tolist(), open(f"{PREDICTIONS_DIR}/{run_id}_seed_{seed}_val_predictions.json", "w"))
    json.dump(y_pred_test.tolist(), open(f"{PREDICTIONS_DIR}/{run_id}_seed_{seed}_test_predictions.json", "w"))

    plot_confusion_matrices(y_true_val, y_pred_val, y_true_test, y_pred_test, run_id, seed)

    val_metrics_list.append(calculate_metrics(y_true_val, y_pred_val))
    test_metrics_list.append(calculate_metrics(y_true_test, y_pred_test))

# ---------------------------------------------------------------------------
# Aggregate training-time (validation) metrics
# ---------------------------------------------------------------------------
metrics = ["accuracy", "f1", "precision", "recall", "f1_macro", "f1_weighted",
           "precision_macro", "precision_weighted", "recall_macro", "recall_weighted"]
final_results = {
    metric: {
        'mean': np.round(np.mean([r[metric] for r in train_time_metrics]), 4),
        'std': np.round(np.std([r[metric] for r in train_time_metrics]), 4)
    } for metric in metrics
}
print(f"[{run_id}] aggregated final results (validation, training-time):")
print(json.dumps(final_results, indent=4))

json_output["aggregated_results"] = final_results
with open(results_file, "a") as f:
    f.write("Aggregated Results Across Runs:\n")
    json.dump(final_results, f, indent=4)
with open(results_file_json, "w") as f:
    json.dump(json_output, f, indent=4)

# ---------------------------------------------------------------------------
# Aggregate val + test metrics across seeds
# ---------------------------------------------------------------------------
val_df = pd.DataFrame(val_metrics_list, index=[f"Seed {s}" for s in SEEDS]).round(4)
test_df = pd.DataFrame(test_metrics_list, index=[f"Seed {s}" for s in SEEDS]).round(4)
val_df.loc["Mean"] = val_df.mean().round(4)
val_df.loc["Std"] = val_df.std().round(4)
test_df.loc["Mean"] = test_df.mean().round(4)
test_df.loc["Std"] = test_df.std().round(4)

val_df.to_csv(f"{PREDICTIONS_DIR}/{run_id}_validation_metrics.csv")
test_df.to_csv(f"{PREDICTIONS_DIR}/{run_id}_test_metrics.csv")

print(f"\n[{run_id}] Validation Metrics")
print(val_df)
print(f"\n[{run_id}] Test Metrics")
print(test_df)
