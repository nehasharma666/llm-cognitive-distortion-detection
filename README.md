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
├── Filter_And_Split_Final_Labels.py
├── Train_And_Evaluate_Binary_Classification.py
├── Train_And_Evaluate_Multiclass_Classification.py
├── Train_And_Evaluate_Multilabel_Classification.py
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

## Data Splits & Filtering

We use `Filter_And_Split_Final_Labels.py` to split each of the 4 `Final_Labels` configuration files into train/dev/test using the fixed Id_Number lists in the `splits` folder (`train_ids.txt`, `dev_ids.txt`, `test_ids.txt`), and then remove rows with an ambiguous label (`Not sure if distortion or not`, `Not sure which distortion`, or `Others`) in either the `RLP_Label` or `MLP_Label` column.

This filtering step is why the LLM-generated configurations end up smaller than the original data:

| Configuration | Train | Dev | Test | Total |
|---|---|---|---|---|
| Original data | 1771 | 379 | 380 | 2530 |
| GPT4-0.5 | 1667 | 346 | 356 | 2369 |
| GPT4-0.7 | 1628 | 344 | 347 | 2319 |
| GPT4o-0.5 | 1494 | 306 | 323 | 2123 |
| GPT4o-0.7 | 1325 | 291 | 272 | 1888 |

## Classification

We fine-tune MentalBERT and MentalRoBERTa on the filtered train/dev/test splits, framed as three separate classification tasks:

- `Train_And_Evaluate_Binary_Classification.py` — Distortion vs. No Distortion
- `Train_And_Evaluate_Multiclass_Classification.py` — single dominant distortion label per input
- `Train_And_Evaluate_Multilabel_Classification.py` — all distortion labels present per input

Each script trains and evaluates across 5 seeds, for a chosen `--model_type` (`bert` or `roberta`) and `--label_type` (`RLP_Label`, `MLP_Label`, or `golden_labels`; multiclass supports `RLP_Label`/`golden_labels` only), and saves trained models, predictions, confusion matrices (binary and multiclass), and aggregated (mean/std) validation and test metrics.


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
## Contact
neha.sharma@ut.ee

https://www.linkedin.com/in/danehasharma/
