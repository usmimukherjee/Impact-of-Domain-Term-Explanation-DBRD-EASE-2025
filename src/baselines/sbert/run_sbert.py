import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

def calculate_dup_indices_with_bug_id_sbert(data, model):
    """
    Calculate the indices of duplicate bug reports using SBERT embeddings.
    """
    embeddings = model.encode(data['Description'].tolist(), convert_to_tensor=True, show_progress_bar=True)

    dup_indices_with_bug_id = {}
    issues = data['Bug ID']

    for i, row in tqdm(data.iterrows(), total=len(data), desc="Processing Bug Reports"):
        issue_id = row['Bug ID']
        dup_id = row['Duplicate_Bug_Ids']

        if dup_id not in issues.values:
            continue

        similarity_scores = util.pytorch_cos_sim(embeddings[i], embeddings).squeeze(0).cpu().numpy()
        sorted_indices = np.argsort(similarity_scores)[::-1][1:]

        dup_bug_index = issues[issues == dup_id].index[0]

        if dup_bug_index in sorted_indices:
            dup_ranking = np.where(sorted_indices == dup_bug_index)[0][0]
        else:
            dup_ranking = -1

        dup_indices_with_bug_id[issue_id] = dup_ranking

    return dup_indices_with_bug_id

def calculate_recall_rates(dup_indices_with_bug_id, thresholds):
    """
    Calculate recall rates at given thresholds.
    """
    counts = {threshold: 0 for threshold in thresholds}

    for value in dup_indices_with_bug_id.values():
        for threshold in thresholds:
            if value != -1 and value < threshold:
                counts[threshold] += 1

    N = len(dup_indices_with_bug_id)
    recall_rates = {}
    for threshold in thresholds:
        recall_rate = counts[threshold] / N
        recall_rates[threshold] = recall_rate

    return recall_rates

def save_recall_rates_to_csv(recall_rates, dataset_name, filepath):
    """
    Save recall rates to a cumulative CSV file.
    """
    recall_data = {"Dataset": dataset_name}
    recall_data.update(recall_rates)

    csv_exists = os.path.exists(filepath)
    
    df = pd.DataFrame([recall_data])
    
    if csv_exists:
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, index=False)
    print(f"Recall rates for {dataset_name} appended to {filepath}")

def data_process(data):
    """
    Preprocess the dataset to clean and prepare for analysis.
    """
    data['Bug ID'] = data['Bug ID'].values.astype('float64')
    nan_value = float("NaN")
    data.replace("", nan_value, inplace=True)
    data['Description'] = data['Description'].values.astype('object')
    data.dropna(subset=["Description"], inplace=True)
    data = data.reset_index(drop=True)
    return data

if __name__ == '__main__':
    filepath = "./data/"
    result_file = ""

    model = SentenceTransformer('all-MiniLM-L6-v2')

    filelist = os.listdir(filepath)
    filelist_sorted = sorted(filelist, reverse=True)

    thresholds = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    for name in filelist_sorted:
        print(f"Working on {name}...")
        base_filename, extension = os.path.splitext(name)

        data = pd.read_csv(os.path.join(filepath, name))
        data = data_process(data)

        dup_indices_with_bug_id = calculate_dup_indices_with_bug_id_sbert(data, model)

        recall_rates = calculate_recall_rates(dup_indices_with_bug_id, thresholds)

        save_recall_rates_to_csv(recall_rates, base_filename, result_file)