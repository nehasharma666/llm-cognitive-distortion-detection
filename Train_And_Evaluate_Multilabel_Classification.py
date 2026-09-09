"""
Train_And_Evaluate_Multilabel_Classification.py — Filter, Split, Train, and Evaluate (Multilabel)

Takes a raw Final_Labels configuration CSV (Id_Number, Patient Question, RLP_Label,
MLP_Label, golden_labels), filters out ambiguous-label rows, drops the negative
"No Distortion" label from the label space (absence = an all-zero prediction, not a
trained class), splits into train/dev/test, fine-tunes a multilabel MentalBERT/
MentalRoBERTa classifier across multiple seeds, and saves self-describing checkpoints
(config.json carries id2label/label2id) ready for inference.

SEEDS — two independent concepts, don't confuse them:
  --seeds             List of model-training seeds (default: the paper's 5 seeds).
                       The SAME train/dev/test split is reused for every seed in this
                       list; only model init / dropout / batch order change. We train
                       once per seed and report mean +/- std, because a single run's
                       score is noisy on its own.
  --random_split_seed Single seed controlling how the DATA is partitioned. Only used
                       when --split_method random. Irrelevant under fixed_ids (the
                       split there comes from the ID files, not a seed).

SPLIT METHOD:
  fixed_ids (default) Reproduces the paper's exact train/dev/test assignment via the
                       ID lists in splits/. Required to get numbers comparable to the
                       paper. The ID -> split assignment is IDENTICAL across all four
                       configs (gpt4-0.5/0.7, gpt4o-0.5/0.7) so cross-config comparison
                       is meaningful — but the row COUNT in each split still differs
                       per config, because each config's own ambiguous-label filtering
                       (above) removes a different subset of rows before this split is
                       applied. Do not expect identical row counts across configs.
  random               Fresh split on this config's filtered data (e.g. for new,
                       unlabeled data with no existing ID lists). NOT comparable to
                       the paper's reported numbers.

EXAMPLE USAGE:

  Full paper-matching run (5 seeds, fixed split):
    python Train_And_Evaluate_Multilabel_Classification.py \
        --config_file Final_Labels/complete_data_gpt4_0.5t.csv \
        --model_type roberta --label_type rlp \
        --split_method fixed_ids \
        --train_ids splits/train_ids.txt --dev_ids splits/dev_ids.txt --test_ids splits/test_ids.txt

  Quick single-seed smoke test, short training, fresh random split:
    python Train_And_Evaluate_Multilabel_Classification.py \
        --config_file Final_Labels/complete_data_gpt4_0.5t.csv \
        --model_type bert --label_type golden \
        --split_method random --seeds 42 --num_train_epochs 3

Outputs (all namespaced by run_id = <config_file basename>_<label_type>_<model_type>,
so runs on different configs never overwrite each other):
  results/results_<run_id>_best/<run_id>_seed_<seed>/   trained model + tokenizer + label_classes.json
  results/<run_id>_results.txt / .json                  per-seed classification reports + aggregated mean/std
  predictions_output/<run_id>_seed_<seed>_{val,test}_predictions.json
  predictions_output/<run_id>_{val,test}_metrics.csv     per-seed + mean/std metrics table
"""

import argparse
import inspect
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
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
    description="Filter, split, train, and evaluate a multilabel cognitive distortion classifier"
)
parser.add_argument('--config_file', type=str, required=True,
                     help='Path to the raw Final_Labels configuration CSV')
parser.add_argument('--model_type', type=str, required=True, choices=['bert', 'roberta'],
                     help='Model type')
parser.add_argument('--label_type', type=str, required=True, choices=['rlp', 'mlp', 'golden'],
                     help='Which label column to train on')
parser.add_argument('--negative_label', type=str, default='No Distortion',
                     help='Label dropped before building the label space (default: "No Distortion")')
parser.add_argument('--ambiguous_labels', type=str, nargs='+',
                     default=['Not sure which distortion', 'Not sure if distortion or not', 'Others'],
                     help='Labels that mark a row as ambiguous; rows with any of these in '
                          'RLP_Label or MLP_Label are dropped entirely')

parser.add_argument('--split_method', type=str, default='fixed_ids', choices=['fixed_ids', 'random'],
                     help="fixed_ids (default): reproduce the paper's exact train/dev/test "
                          "assignment using the ID lists in splits/ — required for results "
                          "comparable to the paper. random: fresh split on this config's "
                          "filtered data (e.g. for new data with no ID lists) — NOT comparable "
                          "to the paper's reported numbers.")
parser.add_argument('--train_ids', type=str, help='Text file, one Id_Number per line (required for --split_method fixed_ids)')
parser.add_argument('--dev_ids', type=str, help='Text file, one Id_Number per line (required for --split_method fixed_ids)')
parser.add_argument('--test_ids', type=str, help='Text file, one Id_Number per line (required for --split_method fixed_ids)')
parser.add_argument('--train_size', type=float, default=0.70, help='Train fraction for --split_method random (default: 0.70)')
parser.add_argument('--dev_size', type=float, default=0.15, help='Dev fraction for --split_method random (default: 0.15)')
parser.add_argument('--test_size', type=float, default=0.15, help='Test fraction for --split_method random (default: 0.15)')
parser.add_argument('--random_split_seed', type=int, default=42,
                     help='Seed controlling the data split itself; only used with --split_method random (default: 42)')

parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024],
                     help='Model-training seeds (default: the paper\'s 5 seeds, to average out '
                          'training-run variance — see module docstring). Pass one value, e.g. '
                          '--seeds 42, for a quick single-run test. Same split used for every seed.')

parser.add_argument('--num_train_epochs', type=int, default=100, help='Max training epochs (default: 100, as used in the paper)')
parser.add_argument('--train_batch_size', type=int, default=16, help='Per-device train batch size (default: 16)')
parser.add_argument('--eval_batch_size', type=int, default=16, help='Per-device eval batch size (default: 16)')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate (default: 5e-5)')
parser.add_argument('--eval_strategy', type=str, default='epoch', choices=['epoch', 'steps'],
                     help='Evaluation/save strategy (default: epoch)')
parser.add_argument('--metric_for_best_model', type=str, default='f1_macro',
                     choices=['f1_macro', 'f1_micro', 'f1_weighted', 'accuracy'],
                     help='Metric used to select the best checkpoint (default: f1_macro)')
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
LABEL_COLUMNS = list(LABEL_COLUMN_MAP.values())
SEEDS = args.seeds

# Fixed 10-class taxonomy (No Distortion excluded — absence is an all-zero
# prediction vector, not a trained class). Kept constant rather than derived
# per-config so every run produces a model with the same label space,
# regardless of which classes happen to survive filtering for a given config.
COGNITIVE_DISTORTION_LABELS = sorted([
    "All or Nothing Thinking",
    "Emotional Reasoning",
    "Fortune Telling",
    "Labeling",
    "Magnification (Catastrophizing)",
    "Mental Filter",
    "Mind Reading",
    "Overgeneralization",
    "Personalization",
    "Should Statements",
])

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
    if not isinstance(labels_list, list):
        return []
    neg_norm = normalize_label(negative_label)
    return [lab for lab in labels_list if normalize_label(lab) != neg_norm]


def load_ids(id_file):
    with open(id_file) as f:
        return {line.strip() for line in f if line.strip()}


def encode_labels(label_lists, mlb):
    return [list(row) for row in mlb.transform(label_lists)]


class PreTokenizedMentalHealthDataset(Dataset):
    def __init__(self, encodings, labels, ids):
        self.encodings = {k: torch.tensor(v) for k, v in encodings.items()}
        self.labels = torch.stack([torch.tensor(label, dtype=torch.float) for label in labels])
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
    predictions = (torch.sigmoid(torch.tensor(pred.predictions)) > 0.5).int()
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_micro": f1_score(labels, predictions, average='micro'),
        "f1_macro": f1_score(labels, predictions, average='macro'),
        "f1_weighted": f1_score(labels, predictions, average='weighted'),
        "precision_micro": precision_score(labels, predictions, average='micro'),
        "precision_macro": precision_score(labels, predictions, average='macro'),
        "precision_weighted": precision_score(labels, predictions, average='weighted'),
        "recall_micro": recall_score(labels, predictions, average='micro'),
        "recall_macro": recall_score(labels, predictions, average='macro'),
        "recall_weighted": recall_score(labels, predictions, average='weighted'),
    }


def calculate_metrics(y_true, y_pred):
    return {
        "f1_micro": f1_score(y_true, y_pred, average='micro'),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
        "precision_micro": precision_score(y_true, y_pred, average='micro'),
        "precision_macro": precision_score(y_true, y_pred, average='macro'),
        "precision_weighted": precision_score(y_true, y_pred, average='weighted'),
        "recall_micro": recall_score(y_true, y_pred, average='micro'),
        "recall_macro": recall_score(y_true, y_pred, average='macro'),
        "recall_weighted": recall_score(y_true, y_pred, average='weighted'),
    }


# ---------------------------------------------------------------------------
# Load + filter the raw configuration file
# ---------------------------------------------------------------------------
data = pd.read_csv(args.config_file)

# float -> int -> str avoids pandas upcasting Id_Number to "12345.0" when the
# column has any missing values, which would silently break every ID match below.
data['Id_Number'] = data['Id_Number'].astype(float).astype(int).astype(str)

ambiguous_norm = {normalize_label(l) for l in args.ambiguous_labels}


def row_is_clean(row):
    # Only RLP_Label / MLP_Label are checked (golden labels are human-annotated
    # and shouldn't contain these LLM hedge phrases). Any match drops the WHOLE
    # row, so rlp/mlp/golden splits always end up with identical row counts.
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

for col in LABEL_COLUMNS:
    data[col] = data[col].apply(parse_labels).apply(lambda x: drop_negative(x, args.negative_label))

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

# ---------------------------------------------------------------------------
# Build the fixed 10-label (CDs only) space, with a sanity check against
# whatever labels actually appear in this config's filtered data
# ---------------------------------------------------------------------------
all_label_lists = pd.concat(
    [train_data[c] for c in LABEL_COLUMNS]
    + [val_data[c] for c in LABEL_COLUMNS]
    + [test_data[c] for c in LABEL_COLUMNS]
)
observed_labels = sorted({lab for row in all_label_lists for lab in row})

missing = set(COGNITIVE_DISTORTION_LABELS) - set(observed_labels)
unexpected = set(observed_labels) - set(COGNITIVE_DISTORTION_LABELS)
if missing:
    print(f"[{run_id}] WARNING: expected CD labels never seen in this config's filtered data: {missing}")
if unexpected:
    print(f"[{run_id}] WARNING: unexpected labels found outside the known taxonomy: {unexpected}")

mlb = MultiLabelBinarizer(classes=COGNITIVE_DISTORTION_LABELS)
mlb.fit([COGNITIVE_DISTORTION_LABELS])

id2label = {i: label for i, label in enumerate(mlb.classes_)}
label2id = {label: i for i, label in enumerate(mlb.classes_)}

source_column = LABEL_COLUMN_MAP[args.label_type]
label_column = f'Label_{args.label_type}'
train_data[label_column] = encode_labels(train_data[source_column], mlb)
val_data[label_column] = encode_labels(val_data[source_column], mlb)
test_data[label_column] = encode_labels(test_data[source_column], mlb)

y_true_val = np.array(val_data[label_column].tolist())
y_true_test = np.array(test_data[label_column].tolist())

# ---------------------------------------------------------------------------
# Tokenize
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[args.model_type], use_auth_token=AUTH_TOKEN)
train_encodings = tokenize_dataset(train_data, tokenizer)
val_encodings = tokenize_dataset(val_data, tokenizer)
test_encodings = tokenize_dataset(test_data, tokenizer)

train_dataset = PreTokenizedMentalHealthDataset(
    train_encodings, train_data[label_column].values, train_data['Id_Number'].astype(int).values
)
val_dataset = PreTokenizedMentalHealthDataset(
    val_encodings, val_data[label_column].values, val_data['Id_Number'].astype(int).values
)
test_dataset = PreTokenizedMentalHealthDataset(
    test_encodings, test_data[label_column].values, test_data['Id_Number'].astype(int).values
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
        num_labels=len(mlb.classes_),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
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
    eval_result = trainer.evaluate()
    print(eval_result, "\n")

    model_path_to_save = f"{RESULTS_DIR}/results_{run_id}_best/{run_id}_seed_{seed}"
    model.save_pretrained(model_path_to_save)
    tokenizer.save_pretrained(model_path_to_save)
    with open(os.path.join(model_path_to_save, "label_classes.json"), "w") as f:
        json.dump(list(mlb.classes_), f, indent=2)

    val_predictions = trainer.predict(val_dataset)
    y_pred_val = (torch.sigmoid(torch.tensor(val_predictions.predictions)) > 0.5).int().numpy()

    with open(results_file, "a") as f:
        f.write(f"Seed {seed} - Classification Report:\n")
        f.write(classification_report(y_true_val, y_pred_val, target_names=mlb.classes_, digits=4) + "\n")
        f.write(f"Subset Accuracy: {eval_result['eval_accuracy']:.4f}\n\n")

    classification_report_dict = classification_report(
        y_true_val, y_pred_val, target_names=mlb.classes_, digits=4, output_dict=True
    )
    json_output[f"seed_{seed}"] = {
        "classification_report": classification_report_dict,
        "subset_accuracy": eval_result['eval_accuracy']
    }

    train_time_metrics.append({
        "f1_micro": classification_report_dict['micro avg']['f1-score'],
        "f1_macro": classification_report_dict['macro avg']['f1-score'],
        "f1_weighted": classification_report_dict['weighted avg']['f1-score'],
        "precision_micro": classification_report_dict['micro avg']['precision'],
        "precision_macro": classification_report_dict['macro avg']['precision'],
        "precision_weighted": classification_report_dict['weighted avg']['precision'],
        "recall_micro": classification_report_dict['micro avg']['recall'],
        "recall_macro": classification_report_dict['macro avg']['recall'],
        "recall_weighted": classification_report_dict['weighted avg']['recall'],
    })

    test_predictions = trainer.predict(test_dataset)
    y_pred_test = (torch.sigmoid(torch.tensor(test_predictions.predictions)) > 0.5).int().numpy()

    json.dump(y_pred_val.tolist(), open(f"{PREDICTIONS_DIR}/{run_id}_seed_{seed}_val_predictions.json", "w"))
    json.dump(y_pred_test.tolist(), open(f"{PREDICTIONS_DIR}/{run_id}_seed_{seed}_test_predictions.json", "w"))

    val_metrics_list.append(calculate_metrics(y_true_val, y_pred_val))
    test_metrics_list.append(calculate_metrics(y_true_test, y_pred_test))

# ---------------------------------------------------------------------------
# Aggregate training-time (validation) metrics
# ---------------------------------------------------------------------------
metrics = ["f1_micro", "f1_macro", "f1_weighted", "precision_micro", "precision_macro",
           "precision_weighted", "recall_micro", "recall_macro", "recall_weighted"]
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
for split, metrics_list in [("val", val_metrics_list), ("test", test_metrics_list)]:
    metrics_df = pd.DataFrame(metrics_list, index=[f"Seed {s}" for s in SEEDS]).round(4)
    metrics_df.loc["Mean"] = metrics_df.mean().round(4)
    metrics_df.loc["Std"] = metrics_df.std().round(4)
    metrics_df.to_csv(f"{PREDICTIONS_DIR}/{run_id}_{split}_metrics.csv")
    print(f"\n[{run_id}] Aggregated {split.capitalize()} Metrics")
    print(metrics_df)
