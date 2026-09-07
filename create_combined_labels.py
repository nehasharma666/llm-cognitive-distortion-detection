"""
create_combined_labels.py — Compute Final Labels and Combine RLP/MLP/golden (original labels from "Detecting Cognitive Distortions 
from Patient-Therapist Interactions" by Sagarika Shreevastava and Peter W. Foltz -)

Cleans "Others" labels and computes a  strict majority-vote Final_Label across the 5
LLM annotation iterations for a given prompt type (RLP or MLP), then combines
the RLP and MLP Final_Labels with the golden (human-annotated) labels into a
single dataset.

Requirements:
    - RLP and MLP annotation CSVs (output of OpenAI_CD_annotations.py) for the
      SAME model and temperature, each with Id_Number, Patient Question, and
      Iter 1..Iter 5 columns
    - Original_data.csv (golden-labeled file) with Id_Number, dominant
      distortion, and Secondary distortion columns

Usage:
    python create_combined_labels.py \
        --RLP_file OpenAI_CD_annotations/results_gpt4_RLP_temp05.csv \
        --MLP_file OpenAI_CD_annotations/results_gpt4_MLP_temp05.csv \
        --golden_file Original_data.csv \
        --output_file Final_Labels/complete_data_gpt4_0.5t.csv

Run once per model x temperature combination (4 times total: gpt4/0.5,
gpt4/0.7, gpt4o/0.5, gpt4o/0.7) to produce the 4 combined label datasets.

Output:
    CSV with columns: Id_Number, Patient Question, RLP_Label, MLP_Label,
    Golden_Labels
"""

import pandas as pd
from collections import Counter
import argparse

annotator_columns = ['Iter 1', 'Iter 2', 'Iter 3', 'Iter 4', 'Iter 5']


# ---------- Step 1: clean labels + compute Final_Label for one file ----------

def remove_others_when_multiple_labels(df, columns):
    def clean_labels(label_list):
        if isinstance(label_list, list):
            if len(set(label_list) - {"Others"}) > 0:
                # Remove "Others" if there's any other label present
                return [label for label in label_list if label != 'Others']
            else:
                # "Others" alone (even if duplicated) -> single "Others"
                return ["Others"] if "Others" in label_list else label_list
        return label_list

    for column in columns:
        df[column] = df[column].apply(clean_labels)
    return df


def max_label_repetition(row):
    all_labels = sum([row[f'Iter {i}'] for i in range(1, 6)], [])
    label_counts = pd.Series(all_labels).value_counts()
    return label_counts.max()


def final_labels(row):
    clean = [label.strip() for i in range(1, 6) for label in row[f'Iter {i}'] if label]
    label_counts = Counter(clean)

    for max_count in [5, 4]:
        most_common = [label for label, count in label_counts.items() if count == max_count]
        if most_common:
            return ', '.join(most_common)

    # Max count is 3 or less
    no_distortion_count = label_counts.get("No Distortion", 0)
    if no_distortion_count >= 1:
        return "Not sure if distortion or not"
    else:
        return "Not sure which distortion"


def compute_final_labels(input_file):
    data = pd.read_csv(input_file)
    data = remove_others_when_multiple_labels(data, annotator_columns)
    data['Max Repetition'] = data.apply(max_label_repetition, axis=1)
    data['Final_Label'] = data.apply(final_labels, axis=1)
    return data


# ---------- Step 2: combine RLP + MLP final labels with golden labels ----------

def combine_golden_labels(df):
    df['Golden_Labels'] = df.apply(
        lambda row: ', '.join(filter(None, [row['dominant distortion'], row['Secondary distortion']])),
        axis=1
    )
    return df[['Id_Number', 'Golden_Labels']]


def main():
    parser = argparse.ArgumentParser(description="Compute final labels and combine RLP, MLP, and golden labels into one file")
    parser.add_argument('--RLP_file', type=str, required=True, help='Raw annotation CSV for the RLP prompt (same model/temperature)')
    parser.add_argument('--MLP_file', type=str, required=True, help='Raw annotation CSV for the MLP prompt (same model/temperature)')
    parser.add_argument('--golden_file', type=str, required=True, help='Original data file with golden (human) labels')
    parser.add_argument('--output_file', type=str, required=True, help='Output CSV file to save combined data')
    args = parser.parse_args()

    RLP_data = compute_final_labels(args.RLP_file)
    MLP_data = compute_final_labels(args.MLP_file)

    if not RLP_data[['Id_Number', 'Patient Question']].equals(MLP_data[['Id_Number', 'Patient Question']]):
        raise ValueError("RLP and MLP files do not have matching Id_Number and Patient Question columns")

    golden_data = pd.read_csv(args.golden_file)
    golden_combined = combine_golden_labels(golden_data)

    combined_data = RLP_data[['Id_Number', 'Patient Question', 'Final_Label']].rename(columns={'Final_Label': 'RLP_Label'})
    combined_data = combined_data.merge(
        MLP_data[['Id_Number', 'Final_Label']].rename(columns={'Final_Label': 'MLP_Label'}),
        on='Id_Number'
    )
    combined_data = combined_data.merge(golden_combined, on='Id_Number')

    combined_data.to_csv(args.output_file, index=False)
    print(f"Combined data saved to {args.output_file}")


if __name__ == '__main__':
    main()