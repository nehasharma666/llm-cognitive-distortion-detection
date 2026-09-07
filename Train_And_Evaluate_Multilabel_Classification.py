"""
Train_And_Evaluate_Multilabel.py — Multilabel Classification Training + Evaluation (5 seeds)

Fine-tunes a MentalBERT/MentalRoBERTa multilabel classifier (each row can carry
zero or more cognitive distortion labels) across 5 seeds, then evaluates each
trained model on the validation and test sets, saving per-seed predictions and
aggregated (mean/std) metrics.

Requirements:
    - train.csv, dev.csv, test.csv, each with Patient Question, Final_Labels_Dom
      (-> RLP_Label), Final_Labels_Non (-> MLP_Label), and golden_labels columns

Usage:
    python Train_And_Evaluate_Multilabel.py --model_type bert --label_type dom

Arguments:
    --model_type   bert or roberta
    --label_type   dom (RLP_Label), non_dom (MLP_Label), or golden (golden_labels)

Output:
    - results/results_<label_type>_<model_type>/seed_<seed>/       training checkpoints
    - results/results_<label_type>_<model_type>_best/seed_<seed>/  saved model + tokenizer
    - results/<label_type>_<model_type>_results.txt / .json        per-seed + aggregated val classification reports
    - predictions_output/..._seed_<seed>_val_predictions.json
    - predictions_output/..._seed_<seed>_test_predictions.json
    - predictions_output/..._val_metrics.csv
    - predictions_output/..._test_metrics.csv
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Train and evaluate multilabel cognitive distortion classifier")
parser.add_argument('--model_type', type=str, required=True, choices=['bert', 'roberta'], help='Model type')
parser.add_argument('--label_type', type=str, required=True, choices=['dom', 'non_dom', 'golden'], help='Label type')
args = parser.parse_args()

MODEL_PATH = {
    'bert': 'mental/mental-bert-base-uncased',
    'roberta': 'mental/mental-roberta-base'
}
AUTH_TOKEN = "your authentication key"
LABEL_COLUMN_MAP = {
    'dom': 'RLP_Label',
    'non_dom': 'MLP_Label',
    'golden': 'golden_labels'
}
SEEDS = [42, 123, 456, 789, 1024]

RESULTS_DIR = "./results"
PREDICTIONS_DIR = "./predictions_output"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


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


def encode_labels(labels, mlb):
    return [list(row) for row in mlb.transform(labels.map(lambda x: x.split(', ') if isinstance(x, str) else []))]


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


# ---------- Load data ----------
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('dev.csv')
test_data = pd.read_csv('test.csv')

# Fit the binarizer once, on the union of ALL splits, so every label seen
# anywhere is represented and the encoding stays consistent train -> eval
all_labels = pd.concat([
    train_data[LABEL_COLUMN_MAP['dom']], train_data[LABEL_COLUMN_MAP['non_dom']], train_data[LABEL_COLUMN_MAP['golden']],
    val_data[LABEL_COLUMN_MAP['dom']], val_data[LABEL_COLUMN_MAP['non_dom']], val_data[LABEL_COLUMN_MAP['golden']],
    test_data[LABEL_COLUMN_MAP['dom']], test_data[LABEL_COLUMN_MAP['non_dom']], test_data[LABEL_COLUMN_MAP['golden']],
])
unique_labels = set(label.strip() for sublist in all_labels.str.split(',') for label in sublist if label)

mlb = MultiLabelBinarizer()
mlb.fit([list(unique_labels)])

label_column = f'Label_{args.label_type.capitalize()}'
train_data[label_column] = encode_labels(train_data[LABEL_COLUMN_MAP[args.label_type]], mlb)
val_data[label_column] = encode_labels(val_data[LABEL_COLUMN_MAP[args.label_type]], mlb)
test_data[label_column] = encode_labels(test_data[LABEL_COLUMN_MAP[args.label_type]], mlb)

y_true_val = np.array(val_data[label_column].tolist())
y_true_test = np.array(test_data[label_column].tolist())

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[args.model_type], use_auth_token=AUTH_TOKEN)
train_encodings = tokenize_dataset(train_data, tokenizer)
val_encodings = tokenize_dataset(val_data, tokenizer)
test_encodings = tokenize_dataset(test_data, tokenizer)

train_dataset = PreTokenizedMentalHealthDataset(train_encodings, train_data[label_column].values, train_data['Id_Number'].values)
val_dataset = PreTokenizedMentalHealthDataset(val_encodings, val_data[label_column].values, val_data['Id_Number'].values)
test_dataset = PreTokenizedMentalHealthDataset(test_encodings, test_data[label_column].values, test_data['Id_Number'].values)


# ---------- Train + evaluate across seeds ----------
results_file = os.path.join(RESULTS_DIR, f"{args.label_type}_{args.model_type}_results.txt")
results_file_json = os.path.join(RESULTS_DIR, f"{args.label_type}_{args.model_type}_results.json")

json_output = {}
train_time_metrics = []
val_metrics_list = []
test_metrics_list = []

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH[args.model_type], num_labels=len(mlb.classes_), use_auth_token=AUTH_TOKEN)
    if torch.cuda.is_available():
        model.cuda()

    training_args = TrainingArguments(
        output_dir=f'{RESULTS_DIR}/results_{args.label_type}_{args.model_type}/{args.label_type}_{args.model_type}_seed_{seed}',
        num_train_epochs=100,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none",
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print(f"----------------{args.label_type}_{args.model_type}-------------")
    print(f"-------------------Training for seed {seed}-------------------\n")
    trainer.train()

    print(f"-------------------Evaluating on validation set for seed {seed}-------------------")
    eval_result = trainer.evaluate()
    print(eval_result, "\n")

    model_path_to_save = f"{RESULTS_DIR}/results_{args.label_type}_{args.model_type}_best/{args.label_type}_{args.model_type}_seed_{seed}"
    model.save_pretrained(model_path_to_save)
    tokenizer.save_pretrained(model_path_to_save)

    # Training-time classification report (validation)
    val_predictions = trainer.predict(val_dataset)
    y_pred_val = (torch.sigmoid(torch.tensor(val_predictions.predictions)) > 0.5).int().numpy()

    with open(results_file, "a") as f:
        f.write(f"Seed {seed} - Classification Report:\n")
        f.write(classification_report(y_true_val, y_pred_val, target_names=mlb.classes_, digits=4) + "\n")
        f.write(f"Subset Accuracy: {eval_result['eval_accuracy']:.4f}\n\n")

    classification_report_dict = classification_report(y_true_val, y_pred_val, target_names=mlb.classes_, digits=4, output_dict=True)
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

    # Evaluate the just-trained model on val + test
    test_predictions = trainer.predict(test_dataset)
    y_pred_test = (torch.sigmoid(torch.tensor(test_predictions.predictions)) > 0.5).int().numpy()

    json.dump(y_pred_val.tolist(), open(f"{PREDICTIONS_DIR}/{args.label_type}_{args.model_type}_seed_{seed}_val_predictions.json", "w"))
    json.dump(y_pred_test.tolist(), open(f"{PREDICTIONS_DIR}/{args.label_type}_{args.model_type}_seed_{seed}_test_predictions.json", "w"))

    val_metrics_list.append(calculate_metrics(y_true_val, y_pred_val))
    test_metrics_list.append(calculate_metrics(y_true_test, y_pred_test))


# ---------- Aggregate training-time (validation) metrics ----------
metrics = ["f1_micro", "f1_macro", "f1_weighted", "precision_micro", "precision_macro",
           "precision_weighted", "recall_micro", "recall_macro", "recall_weighted"]
final_results = {
    metric: {
        'mean': np.round(np.mean([r[metric] for r in train_time_metrics]), 4),
        'std': np.round(np.std([r[metric] for r in train_time_metrics]), 4)
    } for metric in metrics
}
print("aggregated final results (validation, training-time):")
print(json.dumps(final_results, indent=4))

json_output["aggregated_results"] = final_results
with open(results_file, "a") as f:
    f.write("Aggregated Results Across Runs:\n")
    json.dump(final_results, f, indent=4)
with open(results_file_json, "w") as f:
    json.dump(json_output, f, indent=4)


# ---------- Aggregate val + test metrics across seeds ----------
for split, metrics_list in [("val", val_metrics_list), ("test", test_metrics_list)]:
    metrics_df = pd.DataFrame(metrics_list, index=[f"Seed {s}" for s in SEEDS]).round(4)
    metrics_df.loc["Mean"] = metrics_df.mean().round(4)
    metrics_df.loc["Std"] = metrics_df.std().round(4)
    metrics_df.to_csv(f"{PREDICTIONS_DIR}/{args.label_type}_{args.model_type}_{split}_metrics.csv")
    print(f"\nAggregated {split.capitalize()} Metrics for {args.label_type.capitalize()} - {args.model_type.capitalize()}")
    print(metrics_df)