## Data
Please download from this [Google Drive](https://drive.google.com/file/d/1M3DJgL7BIULQtlfaTUahCyVrZz80Ep7j/view?usp=sharing) and put it in the `./data/` directory.

## Experimental Results
Check the results in `./results` directory.

## Commands
### Generate input for REP
To run REP, we need prepare one file for each different json file: `dbrd_test.txt`. Go to `SABD` directory and run the following command:

```bash
python classical_approach/generate_input_dbrd.py --database ../data/keywords/spark/spark_gpt_p1_r1.json --test ../data/raw/test_spark.txt --output ../data/rep/dbrd_test_spark_gpt_p1_r1.txt

python classical_approach/generate_input_dbrd.py --database ../data/keywords/spark/spark_kpminer_test_1.json --test ../data/raw/test_spark.txt --output ../data/rep/dbrd_test_spark_kpminer.txt
```

### Run REP

```bash
build/bin/fast-dbrd -n 20240508_spark_gpt_p1_r1 -r ranknet-configs/full-textual-no-version.cfg --ts /app/tosem-sampel-data/spark/timestamp_file.txt --time-constraint 365 --training-duplicates 273 --recommend /app/tosem-sampel-data/spark/dbrd_test_spark_gpt_p1_r1.txt --trainfile /app/tosem-sampel-data/spark/sampled_training_spark_triplets_random_1.txt

build/bin/fast-dbrd -n 20240510_hadoop_gpt_length_content_r1 -r ranknet-configs/full-textual-no-version.cfg --ts /app/tosem-sampel-data/hadoop/timestamp_file.txt --time-constraint 365 --training-duplicates 285 --recommend /app/tosem-sampel-data/hadoop/dbrd_test_hadoop_gpt_length_content_r${i}.txt --trainfile /app/tosem-sampel-data/hadoop/sampled_training_hadoop_triplets_random_1.txt

build/bin/fast-dbrd -n 20240510_kibana_gpt_p1_r1 -r ranknet-configs/full-textual-no-version.cfg --ts /app/tosem-sampel-data/kibana/timestamp_file.txt --time-constraint 365 --training-duplicates 286 --recommend /app/tosem-sampel-data/kibana/dbrd_test_kibana_gpt_p1_r${i}.txt --trainfile /app/tosem-sampel-data/kibana/training_kibana_triplets_random_1.txt
```

## RQ1: Comparing with Baselines
For SABD and Siamese Pair, we use the implementation in the [replication package](https://github.com/soarsmu/TOSEM-DBRD) of Zhang et al.[1].

### Reference
[1] Zhang, T., Han, D., Vinayakarao, V., Irsan, I. C., Xu, B., Thung, F., ... & Jiang, L. (2023). Duplicate bug report detection: How far are we?. ACM Transactions on Software Engineering and Methodology, 32(4), 1-32.

## RQ2
### RQ2.1: Selection Rules
Preparing the data `src/handle_data.ipynb`

### RQ2.2: ChatGPT Alternatives
Comparing with other keyword extraction methods

- Running LLaMA 3 and Phi-3
path: `src/run_llm.ipynb`

- Running OpenChat
Repo: https://github.com/imoneoi/openchat

```bash
CUDA_VISIBLE_DEVICES=2 python -m ochat.serving.openai_api_server --model openchat/openchat-3.5-0106'
```
We have deplpyed the API server and inference the model with the requests. There are 13 cases returning <Response [400]>, thus, we kept their original content.

- Running TFIDF
path: `src/run_tfidf.ipynb`

- Running KP-Miner
path: `src/run_kpminer.ipynb`

- Running YAKE
path: `src/run_yake.ipynb`


### RQ2.3: Prompt Alternatives
path: `src/run_gpt.ipynb`
