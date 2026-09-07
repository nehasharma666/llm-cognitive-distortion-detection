"""
Filter_And_Split_Final_Labels.py — Train/Dev/Test Splits + Ambiguous-Label Filtering

For each of the 4 Final_Labels configuration files (gpt4/0.5, gpt4/0.7, gpt4o/0.5,
gpt4o/0.7), splits the data into train/dev/test using pre-defined Id_Number lists,
reports how many rows carry an ambiguous or "Others" label, then removes those rows
from each split.

A row is removed if either label column contains any of:
    - "Not sure if distortion or not"
    - "Not sure which distortion"
    - "Others"

Requirements:
    - The 4 Final_Labels CSVs (output of create_combined_Labels.py), each with an
      Id_Number column and two label columns (default names below)
    - Three ID list files (one Id_Number per line) for train, dev, and test

python Filter_And_Split_Final_Labels.py \
    --input_dir Final_Labels \
    --train_ids splits/train_ids.txt \
    --dev_ids splits/dev_ids.txt \
    --test_ids splits/test_ids.txt \
    --output_dir Final_Labels_Filtered \
    --dom_column RLP_Label \
    --non_column MLP_Label

Output:
    - <output_dir>/<config>_train.csv, _dev.csv, _test.csv for each of the 4
      configurations (ambiguous-label rows removed)
    - <output_dir>/split_sizes_summary.csv — Train/Dev/Test/Total row counts per
      configuration (Table 3)
"""

import os
import glob
import argparse
import pandas as pd

TARGET_LABELS = ["Not sure if distortion or not", "Not sure which distortion", "Others"]


def load_ids(id_file):
    with open(id_file) as f:
        return {line.strip() for line in f if line.strip()}


def split_by_ids(df, train_ids, dev_ids, test_ids):
    df_train = df[df['Id_Number'].astype(str).isin(train_ids)].reset_index(drop=True)
    df_dev = df[df['Id_Number'].astype(str).isin(dev_ids)].reset_index(drop=True)
    df_test = df[df['Id_Number'].astype(str).isin(test_ids)].reset_index(drop=True)
    return df_train, df_dev, df_test


def calculate_label_statistics(dfs, df_names, dom_column, non_column, target_labels):
    result = []
    for df, name in zip(dfs, df_names):
        total_rows = len(df)
        total_label_counts = {label: 0 for label in target_labels}

        for _, row in df.iterrows():
            dom_labels = row[dom_column].split(', ') if pd.notna(row[dom_column]) else []
            non_labels = row[non_column].split(', ') if pd.notna(row[non_column]) else []
            for label in target_labels:
                if label in dom_labels or label in non_labels:
                    total_label_counts[label] += 1

        summary = {"Dataset": name, "Total Rows": total_rows}
        for label in target_labels:
            summary[f"{label} (count)"] = total_label_counts[label]
            summary[f"{label} (%)"] = round((total_label_counts[label] / total_rows) * 100, 2) if total_rows else 0
        result.append(summary)

    return pd.DataFrame(result)


def exclude_ambiguous_labels(df, dom_column, non_column, target_labels):
    def row_is_clean(row):
        if pd.isna(row[dom_column]) or pd.isna(row[non_column]):
            return True
        dom_labels = row[dom_column].split(', ')
        non_labels = row[non_column].split(', ')
        return all(label not in dom_labels and label not in non_labels for label in target_labels)

    mask = df.apply(row_is_clean, axis=1)
    return df[mask].reset_index(drop=True)


def process_config(input_file, train_ids, dev_ids, test_ids, dom_column, non_column, output_dir):
    config_name = os.path.splitext(os.path.basename(input_file))[0]
    df = pd.read_csv(input_file)

    df_train, df_dev, df_test = split_by_ids(df, train_ids, dev_ids, test_ids)

    stats = calculate_label_statistics(
        [df_train, df_dev, df_test], ['train', 'dev', 'test'], dom_column, non_column, TARGET_LABELS
    )
    print(f"\n{config_name} — ambiguous label stats before filtering:")
    print(stats)

    filtered_train = exclude_ambiguous_labels(df_train, dom_column, non_column, TARGET_LABELS)
    filtered_dev = exclude_ambiguous_labels(df_dev, dom_column, non_column, TARGET_LABELS)
    filtered_test = exclude_ambiguous_labels(df_test, dom_column, non_column, TARGET_LABELS)

    filtered_train.to_csv(os.path.join(output_dir, f"{config_name}_train.csv"), index=False)
    filtered_dev.to_csv(os.path.join(output_dir, f"{config_name}_dev.csv"), index=False)
    filtered_test.to_csv(os.path.join(output_dir, f"{config_name}_test.csv"), index=False)

    return {
        "Configuration": config_name,
        "Train": len(filtered_train),
        "Dev": len(filtered_dev),
        "Test": len(filtered_test),
        "Total": len(filtered_train) + len(filtered_dev) + len(filtered_test),
    }


def main():
    parser = argparse.ArgumentParser(description="Split Final Labels into train/dev/test and remove ambiguous-label rows")
    parser.add_argument('--input_dir', type=str, required=True, help='Folder containing the 4 Final_Labels CSVs')
    parser.add_argument('--train_ids', type=str, required=True, help='Text file with one Id_Number per line for the train split')
    parser.add_argument('--dev_ids', type=str, required=True, help='Text file with one Id_Number per line for the dev split')
    parser.add_argument('--test_ids', type=str, required=True, help='Text file with one Id_Number per line for the test split')
    parser.add_argument('--output_dir', type=str, required=True, help='Folder to save filtered train/dev/test CSVs and the summary table')
    parser.add_argument('--dom_column', type=str, default='RLP_Label', help='Column with the dominant-label prompt output (default: Final_Labels_Dom)')
    parser.add_argument('--non_column', type=str, default='MLP_Label', help='Column with the non-dominant/multi-label prompt output (default: Final_Labels_Non)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_ids = load_ids(args.train_ids)
    dev_ids = load_ids(args.dev_ids)
    test_ids = load_ids(args.test_ids)

    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    summary_rows = [
        process_config(f, train_ids, dev_ids, test_ids, args.dom_column, args.non_column, args.output_dir)
        for f in input_files
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.output_dir, "split_sizes_summary.csv"), index=False)
    print("\nSplit sizes summary:")
    print(summary_df)


if __name__ == '__main__':
    main()