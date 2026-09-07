"""
Train_And_Evaluate_Binary.py — Binary Classification Training + Evaluation (5 seeds)

Fine-tunes a MentalBERT/MentalRoBERTa binary classifier (Distortion vs. No
Distortion) across 5 seeds, then evaluates each trained model on the
validation and test sets, saving per-seed predictions, confusion matrices,
and aggregated (mean/std) metrics.

Requirements:
    - train.csv, dev.csv, test.csv, each with a Patient Question column and
      the label column for the chosen --label_type
    - HuggingFace auth token for mental/mental-bert-base-uncased and
      mental/mental-roberta-base

Usage:
    python Train_And_Evaluate_Binary.py --model_type bert --label_type dom

Arguments:
    --model_type   bert or roberta
    --label_type   dom (RLP_Label), non_dom (MLP_Label), or golden (Golden_Labels)

Output:
    - results/results_<label_type>_<model_type>/seed_<seed>/       training checkpoints
    - results/results_<label_type>_<model_type>_best/seed_<seed>/  saved model + tokenizer
    - results/<label_type>_<model_type>_results.txt / .json        per-seed + aggregated val classification reports
    - predictions_output/..._seed_<seed>_val_predictions.json
    - predictions_output/..._seed_<seed>_test_predictions.json
    - predictions_output/..._seed_<seed>_confusion_matrix.png
    - predictions_output/..._validation_metrics.csv
    - predictions_output/..._test_metrics.csv
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Train and evaluate binary cognitive distortion classifier")
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
    def __init__(self, encodings, labels, ids=None):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.ids = ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        if self.ids is not None:
            item['id'] = torch.tensor(self.ids[idx], dtype=torch.long)
        return item


def tokenize_dataset(dataset, tokenizer):
    return tokenizer(
        dataset["Patient Question"].tolist(),
        padding="max_length", truncation=True, max_length=512, return_tensors="pt"
    )


def apply_binary_labels(data, label_type):
    column = LABEL_COLUMN_MAP[label_type]
    return data[column].apply(lambda x: 0 if 'No Distortion' in x else 1).values


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average='macro'),
        "recall_macro": recall_score(y_true, y_pred, average='macro'),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "precision_weighted": precision_score(y_true, y_pred, average='weighted'),
        "recall_weighted": recall_score(y_true, y_pred, average='weighted'),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
    }


def plot_confusion_matrices(y_true_val, y_pred_val, y_true_test, y_pred_test, label_type, model_type, seed):
    display_labels = ["No Distortion", "Distortion"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Confusion Matrices for {label_type.capitalize()} - {model_type.capitalize()} (Seed {seed})")

    cm_val = confusion_matrix(y_true_val, y_pred_val)
    ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=display_labels).plot(ax=axes[0], cmap='Blues')
    axes[0].set_title("Validation Set")

    cm_test = confusion_matrix(y_true_test, y_pred_test)
    ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=display_labels).plot(ax=axes[1], cmap='Blues')
    axes[1].set_title("Test Set")

    plt.tight_layout()
    plt.savefig(f"{PREDICTIONS_DIR}/{label_type}_{model_type}_seed_{seed}_confusion_matrix.png")
    plt.close()


# ---------- Load data ----------
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('dev.csv')
test_data = pd.read_csv('test.csv')

label_column = f'Binary_Label_{args.label_type.capitalize()}'
train_data[label_column] = apply_binary_labels(train_data, args.label_type)
val_data[label_column] = apply_binary_labels(val_data, args.label_type)
test_data[label_column] = apply_binary_labels(test_data, args.label_type)

y_true_val = val_data[label_column].values
y_true_test = test_data[label_column].values

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

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH[args.model_type], num_labels=2, use_auth_token=AUTH_TOKEN)
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
        metric_for_best_model="f1",
        report_to="none",
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda pred: {
            "accuracy": accuracy_score(pred.label_ids, pred.predictions.argmax(-1)),
            "precision": precision_score(pred.label_ids, pred.predictions.argmax(-1), average='binary'),
            "recall": recall_score(pred.label_ids, pred.predictions.argmax(-1), average='binary'),
            "f1": f1_score(pred.label_ids, pred.predictions.argmax(-1), average='binary')
        },
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print(f"----------------{args.label_type}_{args.model_type}-------------")
    print(f"-------------------Training for seed {seed}-------------------\n")
    trainer.train()

    print(f"-------------------Evaluating on validation set for seed {seed}-------------------")
    print(trainer.evaluate(), "\n")

    model_path_to_save = f"{RESULTS_DIR}/results_{args.label_type}_{args.model_type}_best/{args.label_type}_{args.model_type}_seed_{seed}"
    model.save_pretrained(model_path_to_save)
    tokenizer.save_pretrained(model_path_to_save)

    # Training-time classification report (validation)
    val_predictions = trainer.predict(val_dataset)
    y_pred_val = np.argmax(val_predictions.predictions, axis=1)

    with open(results_file, "a") as f:
        f.write(f"Seed {seed} - Classification Report:\n")
        f.write(classification_report(y_true_val, y_pred_val, digits=4) + "\n")

    classification_report_dict = classification_report(y_true_val, y_pred_val, digits=4, output_dict=True)
    json_output[f"seed_{seed}"] = {"classification_report": classification_report_dict}

    train_time_metrics.append({
        "accuracy": classification_report_dict['accuracy'],
        "f1_macro": classification_report_dict['macro avg']['f1-score'],
        "f1_weighted": classification_report_dict['weighted avg']['f1-score'],
        "precision_macro": classification_report_dict['macro avg']['precision'],
        "precision_weighted": classification_report_dict['weighted avg']['precision'],
        "recall_macro": classification_report_dict['macro avg']['recall'],
        "recall_weighted": classification_report_dict['weighted avg']['recall'],
    })

    # Evaluate the just-trained model on val + test
    test_predictions = trainer.predict(test_dataset)
    y_pred_test = np.argmax(test_predictions.predictions, axis=1)

    json.dump(y_pred_val.tolist(), open(f"{PREDICTIONS_DIR}/{args.label_type}_{args.model_type}_seed_{seed}_val_predictions.json", "w"))
    json.dump(y_pred_test.tolist(), open(f"{PREDICTIONS_DIR}/{args.label_type}_{args.model_type}_seed_{seed}_test_predictions.json", "w"))

    plot_confusion_matrices(y_true_val, y_pred_val, y_true_test, y_pred_test, args.label_type, args.model_type, seed)

    val_metrics_list.append(calculate_metrics(y_true_val, y_pred_val))
    test_metrics_list.append(calculate_metrics(y_true_test, y_pred_test))


# ---------- Aggregate training-time (validation) metrics ----------
metrics = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "precision_weighted", "recall_macro", "recall_weighted"]
final_results = {
    metric: {
        'mean': np.round(np.mean([r[metric] for r in train_time_metrics]), 4),
        'std': np.round(np.std([r[metric] for r in train_time_metrics]), 4)
    } for metric in metrics
}
print("Final_aggregated_results (validation, training-time):")
print(json.dumps(final_results, indent=4))

json_output["aggregated_results"] = final_results
with open(results_file, "a") as f:
    f.write("Aggregated Results Across Runs:\n")
    json.dump(final_results, f, indent=4)
with open(results_file_json, "w") as f:
    json.dump(json_output, f, indent=4)


# ---------- Aggregate val + test metrics across seeds ----------
val_df = pd.DataFrame(val_metrics_list, index=[f"Seed {s}" for s in SEEDS]).round(4)
test_df = pd.DataFrame(test_metrics_list, index=[f"Seed {s}" for s in SEEDS]).round(4)
val_df.loc["Mean"] = val_df.mean().round(4)
val_df.loc["Std"] = val_df.std().round(4)
test_df.loc["Mean"] = test_df.mean().round(4)
test_df.loc["Std"] = test_df.std().round(4)

val_df.to_csv(f"{PREDICTIONS_DIR}/{args.label_type}_{args.model_type}_validation_metrics.csv")
test_df.to_csv(f"{PREDICTIONS_DIR}/{args.label_type}_{args.model_type}_test_metrics.csv")

print(f"\nValidation Metrics for {args.label_type.capitalize()} - {args.model_type.capitalize()}")
print(val_df)
print(f"\nTest Metrics for {args.label_type.capitalize()} - {args.model_type.capitalize()}")
print(test_df)