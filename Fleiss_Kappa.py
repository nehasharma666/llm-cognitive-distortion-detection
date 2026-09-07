"""
Fleiss_Kappa.py — Inter-run Agreement via Fleiss' Kappa

Computes per-label Fleiss' Kappa scores across the 5 LLM annotation iterations
(Iter 1..Iter 5) in a single annotation results file, measuring how consistently
the model assigns each cognitive distortion label across repeated runs.

Requirements:
    - An annotation results CSV (output of OpenAI_CD_annotations.py) containing
      Id_Number and Iter 1..Iter 5 columns

Usage:
    python Fleiss_Kappa.py --input_file <path_to_annotations_csv> --output_file <path_to_scores_csv>

Arguments:
    --input_file    one of the 8 annotation CSVs from OpenAI_CD_annotations.py
                     (one model x prompt x temperature combination)
    --output_file   where to save the per-label Fleiss' Kappa scores

Run once per input file (8 times total) to reproduce all scores; the per-file
outputs were then compiled by hand into Fleiss_Kappa_Scores.csv.

Output:
    CSV with columns Label, Fleiss_Kappa, plus a console printout of the
    average Kappa across all labels and across the 10 predefined cognitive
    distortion labels only.
"""

import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Calculate Fleiss' Kappa for Annotation Consistency")
parser.add_argument('--input_file', type=str, required=True, help='Input CSV file containing annotation iterations i.e. 8 files from prompt type, gpt model type and temperature type')
parser.add_argument('--output_file', type=str, required=True, help='Output CSV file to save Fleiss Kappa scores')
args = parser.parse_args()

# Load data
data = pd.read_csv(args.input_file)
annotator_columns = ['Iter 1', 'Iter 2', 'Iter 3', 'Iter 4', 'Iter 5']

#Predefined labels for cognitive distortions
cognitive_distortions = [
    "Emotional Reasoning", "Overgeneralization", "Mental Filter", "Should Statements", "All or Nothing Thinking", 
    "Mind Reading", "Fortune Telling", "Magnification (Catastrophizing)", "Personalization", "Labeling", "No Distortion"
]


def extract_unique_labels(data, columns):
    unique_labels = set()
    for col in columns:
        data[col].apply(lambda labels: unique_labels.update(labels.split(', ')))
    return unique_labels

def create_binary_dataframes(data, unique_labels):
    binary_dataframes = {}
    # Function to evaluate and return the presence (1) or absence (0) of each label in each iteration
    def evaluate_presence(row, label):
        return [int(label in iteration.split(', ')) for iteration in row]
    
    # For each unique label, create a binary dataframe
    for label in unique_labels:
        binary_df = data.apply(
            lambda row: evaluate_presence(row[annotator_columns], label),
            axis=1, result_type='expand'
        )
        binary_df.columns = annotator_columns
        binary_df['Id_Number'] = data['Id_Number']
        binary_dataframes[label] = binary_df

    return binary_dataframes

def calculate_fleiss_kappa_for_label(binary_df):
    contingency_table = binary_df[annotator_columns].apply(pd.Series.value_counts, axis=1).reindex(columns=[0, 1], fill_value=0)
    fleiss_kappa_score = fleiss_kappa(contingency_table.values)
    return fleiss_kappa_score

# Extract unique labels from data
unique_labels = extract_unique_labels(data, annotator_columns)

# Create binary dataframes for each label
binary_dataframes = create_binary_dataframes(data, unique_labels)

# Calculate Fleiss' Kappa for each label
fleiss_kappa_scores = {
    label: calculate_fleiss_kappa_for_label(binary_df)
    for label, binary_df in binary_dataframes.items()
}

# Prepare results
scores_df = pd.DataFrame(
    list(fleiss_kappa_scores.items()), 
    columns=['Label', 'Fleiss_Kappa']
).sort_values(by='Label').reset_index(drop=True)

# Save results
scores_df.to_csv(args.output_file, index=False)

print(f"Fleiss' Kappa scores saved to {args.output_file}")

# Average Fleiss's Kappa score for all the labels
average_kappa_all_labels = np.round(scores_df['Fleiss_Kappa'].mean(),4)
print("Average kappa score:",average_kappa_all_labels)

# Calculate average Fleiss' Kappa for the predefined cognitive distortions only
average_kappa_cogntive_distortion_list_only = np.round(scores_df[scores_df['Label'].isin(cognitive_distortions)]['Fleiss_Kappa'].mean(), 4)
print("Average Kappa Score for Cognitive Distortions:",average_kappa_cogntive_distortion_list_only)

