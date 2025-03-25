import pandas as pd
import re
import string
import os
import warnings
from rank_bm25 import BM25Okapi
from nltk import word_tokenize
import numpy as np
import json
from tqdm import tqdm
warnings.filterwarnings("ignore")

def tokenizing_corpus(corpus):
    tokens_from_corpus = []
    for s in corpus:
        cleanedTex = re.sub(r'[^\w\s]', '', str(s)).lower()
        words = (word_tokenize(cleanedTex))
        tokens_from_corpus.append(words)
    # print(tokens_generated)
    return tokens_from_corpus

def tokenizing_query(sentence):
    tokens_from_query = []
    cleanedTex=re.sub(r'[^\w\s]','',str(sentence)).lower()
    words = (word_tokenize(cleanedTex))
    tokens_from_query.append(words)
    tokens_from_query = tokens_from_query[0]
    # print(tokens_generated)
    return tokens_from_query

def calculate_dup_indices_with_bug_id(data):
       
    tokens = tokenizing_corpus(data.Description)
    bm25 = BM25Okapi(tokens)
    count = 0
    dup_indices_with_bug_id = {}
    issues = data['Bug ID']
    
    for i, row in tqdm(data.iterrows(), total=len(data)):
        
        issue_id = row['Bug ID']
        dup_id = row.Duplicate_Bug_Ids
        
        if dup_id not in issues.values:
            count += 1
            continue

        # Get BM25 similarity scores for the current tokens
        similarity_scores = bm25.get_scores(tokens[i])

        # Sort indices of scores in descending order, excluding the first element (self-match)
        sorted_indices = np.argsort(similarity_scores)[::-1][1:]

        # Find the index of the duplicate bug ID within the issues array
        dup_bug_index = issues[issues == dup_id].index[0]

                # Check if the dup_bug_index exists in sorted_indices
        if dup_bug_index in sorted_indices:
            # Get the ranking position of the duplicate bug index
            dup_ranking = np.where(sorted_indices == dup_bug_index)[0][0]
        else:
            # If the duplicate bug ID isn't found, assign a default value
            dup_ranking = -1  # You can use -1 or None to indicate "not found"

        # Store the ranking position in the dictionary with the Bug ID as the key
        dup_indices_with_bug_id[issue_id] = dup_ranking
    
    return dup_indices_with_bug_id, count


def calculate_recall_rates(dup_indices_with_bug_id, thresholds=[1, 5, 10, 100]):
    counts = {threshold: 0 for threshold in thresholds}

    for value in dup_indices_with_bug_id.values():
        for threshold in thresholds:
            if value < threshold:
                counts[threshold] += 1

    N = len(dup_indices_with_bug_id)
    recall_rates = {}
    for threshold in thresholds:
        recall_rate = counts[threshold] / N
        recall_rates[threshold] = recall_rate
        print(f"count at {threshold}: {counts[threshold]}")
        print(f"recall_rate_at_{threshold}: {recall_rate:.2f}")
    
    return recall_rates

def save_recall_rates(recall_rates, filepath):
    # Save the recall rates as a JSON file
    with open(filepath, 'w') as f:
        json.dump(recall_rates, f, indent=4)
    print(f"Recall rates saved to {filepath}")

def data_process(data):
    data['Bug ID']= data['Bug ID'].values.astype('float64')
    nan_value = float("NaN")
    data.replace("", nan_value, inplace=True)
    data['Description']=data['Description'].values.astype('object')
    data.dropna(subset = ["Description"], inplace=True)
    cond = (data['Description'] == 'NaN')
    cond.unique()
    data = data.reset_index(drop=True)
    return data

if __name__ == '__main__':
    filepath = ""
    result_path = ""
    
    filelist = os.listdir(filepath)
    filelist_sorted = sorted(filelist, reverse=True)
    for name in filelist_sorted:
        print("working on --------------"+name+"----------------------")
        base_filename, extension = os.path.splitext(name)
        directory_path = result_path + base_filename
        if not os.path.isdir(directory_path):
            os.makedirs(directory_path)
        
        data = pd.read_csv(filepath+name)
        data = data_process(data)
        dup_indices_with_bug_id, count = calculate_dup_indices_with_bug_id(data)
        recall_rates = calculate_recall_rates(dup_indices_with_bug_id, thresholds=[1, 5, 10, 100])
        recall_rates_filepath = os.path.join(directory_path, "")
        save_recall_rates(recall_rates, recall_rates_filepath)
