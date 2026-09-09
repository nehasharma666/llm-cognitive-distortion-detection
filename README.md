# Towards Consistent Detection of Cognitive Distortions: LLM-Based Annotation and Dataset-Agnostic Evaluation

Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)

DOI: [10.63317/4joqxtgdgxq2](https://doi.org/10.63317/4joqxtgdgxq2)

## Overview

This repository contains the data, annotations, and scripts for our LREC 2026 paper.

## Repository Structure

```
.
├── Original_data.csv
├── OpenAI_CD_annotations.py
├── annotate_job.job
├── OpenAI_CD_annotations/
├── Fleiss_Kappa.py
├── Fleiss_Kappa_Scores.csv
├── create_combined_labels.py
├── Final_Labels/
├── splits/
├── Train_And_Evaluate_Binary_Classification.py
├── Train_And_Evaluate_Multiclass_Classification.py
├── Train_And_Evaluate_Multilabel_Classification.py
├── predict_binary.py
├── predict_multiclass.py
├── predict_multilabel.py
└── README.md
```

## Data

We use `Original_data.csv`, which contains the `Id_Number` and `Patient Question` columns.

This is the link of the data: [Therapist Q&A](https://www.kaggle.com/datasets/arnmaud/therapist-qa)

## Annotation

We use `OpenAI_CD_annotations.py` and `annotate_job.job` to get annotations from the GPT-4 and GPT-4o models, at temperatures 0.5 and 0.7, using two prompts: RLP and MLP. The resulting files are saved into the `OpenAI_CD_annotations` folder.

## Inter-Annotator Agreement (Fleiss' Kappa)

We use `Fleiss_Kappa.py` to calculate per-label Fleiss' Kappa scores, measuring agreement across the 5 GPT annotation iterations for each of the 8 model/prompt/temperature configurations.

The compiled Kappa scores (dominant vs. non-dominant distortion, across all 8 configurations) are in `Fleiss_Kappa_Scores.csv`.

## Final Labels

We use `create_combined_labels.py` to compute a majority-vote Final_Label across the 5 annotation iterations for each prompt type (RLP and MLP), then combine the RLP and MLP labels with the golden (human-annotated) labels into one file per model/temperature configuration.

Run once per model x temperature combination (4 times total: gpt4/0.5, gpt4/0.7, gpt4o/0.5, gpt4o/0.7). Each resulting file, saved in the `Final_Labels` folder, has: `Id_Number`, `Patient Question`, `RLP_Label`, `MLP_Label`, and `Golden_Labels`.

The annotations, Fleiss' Kappa scores, and final labels above are all provided in this repository, so you can inspect them directly and choose whichever model/prompt/temperature configuration fits your use case without re-running anything. The remaining sections below document the process for anyone who wants to replicate the full pipeline from that point onward.

## Classification

Each of the three classification scripts below takes a raw `Final_Labels` configuration CSV directly (not a pre-split train/dev/test set) and handles filtering, splitting, training, and evaluation internally:

- `Train_And_Evaluate_Binary_Classification.py` — Distortion vs. No Distortion
- `Train_And_Evaluate_Multiclass_Classification.py` — single dominant distortion label per input (11 classes: the 10 CDs + "No Distortion" as an explicit class)
- `Train_And_Evaluate_Multilabel_Classification.py` — all distortion labels present per input (10 CD classes; "No Distortion" is not a trained class — its absence is represented as an all-zero prediction, since independent per-label predictions could otherwise contradict each other, e.g. "No Distortion" and "Overgeneralization" both predicted present at once)

Each script:
1. Reads the given `--config_file` (one of the 4 files in `Final_Labels/`) and removes rows with an ambiguous label (`Not sure if distortion or not`, `Not sure which distortion`, or `Others`) in either `RLP_Label` or `MLP_Label`. This filtering is why the LLM-generated configurations end up smaller than the original data — see the table below.
2. Splits into train/dev/test, either using the fixed Id_Number lists in `splits/` (`--split_method fixed_ids`, the default — reproduces the paper's exact split and is required for results comparable to the paper), or a fresh split on the filtered data (`--split_method random` — for new/unlabeled data with no existing ID lists; not paper-comparable).
3. Trains and evaluates for a chosen `--model_type` (`bert` or `roberta`) and `--label_type` (`rlp`, `mlp`, or `golden`; multiclass supports `rlp`/`golden` only — MLP rows can carry many labels, making "first label = dominant" too weak an approximation for a single-label task).
4. Trains across multiple seeds (`--seeds`, default the paper's 5 seeds — `42 123 456 789 1024`), saving trained models, predictions, confusion matrices (binary and multiclass), and aggregated (mean/std) validation and test metrics.

All hyperparameters used in the paper (epochs, batch size, learning rate, evaluation strategy, best-model metric, early stopping patience) are exposed as CLI arguments with the paper's values as defaults, so they can be overridden without editing the script.

Each script's own docstring documents its full argument list and gives two example commands: one that reproduces the paper's setup (5 seeds, fixed splits), and one quick single-seed smoke test (`--seeds 42`, a random split, fewer epochs) for checking that everything runs before committing to a full run.

Resulting split sizes after ambiguous-label filtering (identical Id_Number → split assignment across all 4 configurations under `--split_method fixed_ids`, though row counts differ per configuration since each one's filtering removes a different subset of rows):

| Configuration | Train | Dev | Test | Total |
|---|---|---|---|---|
| Original data | 1771 | 379 | 380 | 2530 |
| GPT4-0.5 | 1667 | 346 | 356 | 2369 |
| GPT4-0.7 | 1628 | 344 | 347 | 2319 |
| GPT4o-0.5 | 1494 | 306 | 323 | 2123 |
| GPT4o-0.7 | 1325 | 291 | 272 | 1888 |

### Prediction

Each classification script saves a self-describing checkpoint — label names are baked into the saved model's `config.json` (`id2label`/`label2id`), with a `label_classes.json` backup in the same folder — so predicting on new data doesn't require separately tracking a label list:

- `predict_binary.py` — Distortion vs. No Distortion, with a confidence score
- `predict_multiclass.py` — single predicted label (of 11), with a confidence score
- `predict_multilabel.py` — all predicted labels present (sigmoid threshold, default 0.5), or "No Distortion" if none exceed the threshold

Each takes `--model_path` (a saved checkpoint folder) and `--input_file` (a CSV with a text column), and writes predictions plus per-label probabilities to `--output_file`.

## Citation

```bibtex
@inproceedings{sharma-etal-2026-consistent,
  title = {Towards Consistent Detection of Cognitive Distortions: LLM-Based Annotation and Dataset-Agnostic Evaluation},
  author = {Sharma, Neha and Agarwal, Navneet and Sirts, Kairit},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {10866--10882},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  doi = {10.63317/4joqxtgdgxq2},
  abstract = {Text-based automated Cognitive Distortion detection is a challenging task due to its subjective nature, with low agreement scores observed even among expert human annotators, leading to unreliable annotations. We explore the use of Large Language Models (LLMs) as consistent and reliable annotators, and propose that multiple independent LLM runs can reveal stable labeling patterns despite the inherent subjectivity of the task. Furthermore, to fairly compare models trained on datasets with different characteristics, we introduce a dataset-agnostic evaluation framework using Cohen's kappa as an effect size measure. This methodology allows for fair cross-dataset and cross-study comparisons where traditional metrics like F1 score fall short. Our results show that GPT-4 can produce consistent annotations (Fleiss's Kappa = 0.78), resulting in improved test set performance for models trained on these annotations compared to those trained on human-labeled data. While human expert verification was inconclusive on our target dataset, our findings suggest that LLMs can offer a scalable and internally consistent alternative for generating training data that supports strong downstream performance in subjective NLP tasks.}
}
```


Note: the underlying Therapist Q&A dataset (`Original_data.csv`) is sourced from [Kaggle](https://www.kaggle.com/datasets/arnmaud/therapist-qa) and remains subject to its own license/terms — this repository's license covers only the code and annotations we produced.

## Contact

Email: neha.sharma@ut.ee

LinkedIn: [linkedin.com/in/danehasharma](https://www.linkedin.com/in/danehasharma)
