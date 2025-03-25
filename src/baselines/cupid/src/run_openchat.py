import requests
import pandas as pd
import os
from tqdm import tqdm
from icecream import ic
from transformers import pipeline
import torch

prompt_template = "Identify keywords from the summary and description of the bug report that can be used to detect duplicates.\n\nOutput format:\nSummary: [Selected Keywords]\nDescription: [Selected Keywords]\n\nSummary: {}\nDescription: {}\n\n"
project = 'spark'
df = pd.read_csv('../data/raw/test_{}.csv'.format(project))
flag_content_df = pd.read_csv(f'../data/ablation/test_{project}_flag_content.csv')


def infer_with_API():
    url = 'http://localhost:18888/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json'
    }

    for run in range(1, 6):
        openchat_folder = f'../data/keywords/{project}/openchat/run_{run}'
        
        if not os.path.exists(openchat_folder.format(project, run)):
            os.makedirs(openchat_folder.format(project, run))

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            bug_id = row['bug_id']
            
            if flag_content_df[flag_content_df['bug_id'] == bug_id]['run_flag'].values[0] == 0:
                continue
            
            if os.path.exists(os.path.join(openchat_folder.format(project, run), f'{bug_id}.txt')):
                continue
            try:
                data = {
                    "model": "openchat_3.5",
                    "messages": [{"role": "user", "content": prompt_template.format(row['short_desc'], row['description'])}],
                    "temperature": 0,
                    "top_p": 1,
                    "max_new_tokens":2048,
                    "frequency_penalty":0,
                    "presence_penalty":0,
                    "seed":42
                }
                # print(requests.post(url, headers=headers, json=data))
                response = requests.post(url, headers=headers, json=data).json()['choices'][0]['message']['content']
            except Exception as e:
                ic(requests.post(url, headers=headers, json=data))
                continue
                # # ic(prompt_template.format(row['short_desc'], row['description'][:1000]))
                # data = [
                #     {
                #         "model": "openchat_3.5",
                #         "messages": [{"role": "user", "content": prompt_template.format(row['short_desc'], row['description'][:1000])}],
                #         "temperature": 0,
                #         "top_p": 1,
                #         "max_new_tokens":2048,
                #         "frequency_penalty":0,
                #         "presence_penalty":0,
                #         "seed":42
                #     }
                # ]
                # # print(requests.post(url, headers=headers, json=data))
                # response = requests.post(url, headers=headers, json=data).json()['choices'][0]['message']['content']
                
            with open(os.path.join(openchat_folder.format(project, run), f'{bug_id}.txt'), 'w') as f:
                f.write(prompt_template.format(row['short_desc'], row['description']))
                f.write('\n\n>>>>>> Response:\n\n')
                f.write(response)

if __name__ == '__main__':
    infer_with_API()