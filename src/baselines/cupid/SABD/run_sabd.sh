# for i in {1..5}
# do
#     python classical_approach/generate_input_dbrd.py --database "../data/ablation/spark_gpt_concise_content_r${i}.json" --test ../data/raw/test_spark.txt --output "../data/rep/dbrd_test_spark_gpt_concise_r${i}.txt"
    # python classical_approach/generate_input_dbrd.py --database "../data/ablation/spark_gpt_length_r${i}.json" --test ../data/raw/test_spark.txt --output "../data/rep/dbrd_test_spark_gpt_length_r${i}.txt"
    # python classical_approach/generate_input_dbrd.py --database "../data/ablation/spark_gpt_content_r${i}.json" --test ../data/raw/test_spark.txt --output "../data/rep/dbrd_test_spark_gpt_content_r${i}.txt"
    # python classical_approach/generate_input_dbrd.py --database "../data/ablation/spark_gpt_length_content_r${i}.json" --test ../data/raw/test_spark.txt --output "../data/rep/dbrd_test_spark_gpt_length_content_r${i}.txt"
    # python classical_approach/generate_input_dbrd.py --database "../data/ablation/hadoop_gpt_length_r${i}.json" --test ../data/raw/test_hadoop.txt --output "../data/rep/dbrd_test_hadoop_gpt_length_r${i}.txt"
    # python classical_approach/generate_input_dbrd.py --database "../data/ablation/hadoop_gpt_content_r${i}.json" --test ../data/raw/test_hadoop.txt --output "../data/rep/dbrd_test_hadoop_gpt_content_r${i}.txt"
    # python classical_approach/generate_input_dbrd.py --database "../data/ablation/hadoop_gpt_length_content_r${i}.json" --test ../data/raw/test_hadoop.txt --output "../data/rep/dbrd_test_hadoop_gpt_length_content_r${i}.txt"
    # python classical_approach/generate_input_dbrd.py --database "../data/ablation/kibana_gpt_length_r${i}.json" --test ../data/raw/test_kibana.txt --output "../data/rep/dbrd_test_kibana_gpt_length_r${i}.txt"
    # python classical_approach/generate_input_dbrd.py --database "../data/ablation/kibana_gpt_content_r${i}.json" --test ../data/raw/test_kibana.txt --output "../data/rep/dbrd_test_kibana_gpt_content_r${i}.txt"
    # python classical_approach/generate_input_dbrd.py --database "../data/ablation/kibana_gpt_length_content_r${i}.json" --test ../data/raw/test_kibana.txt --output "../data/rep/dbrd_test_kibana_gpt_length_content_r${i}.txt"
    # python classical_approach/generate_input_dbrd.py --database "../data/keywords/hadoop/hadoop_gpt_p1_r${i}.json" --test ../data/raw/test_hadoop.txt --output "../data/rep/dbrd_test_hadoop_gpt_p1_r${i}.txt"
    # python classical_approach/generate_input_dbrd.py --database "../data/keywords/kibana/kibana_gpt_p1_r${i}.json" --test ../data/raw/test_kibana.txt --output "../data/rep/dbrd_test_kibana_gpt_p1_r${i}.txt"
# done

# python classical_approach/generate_input_dbrd.py --database "../data/ablation/spark_llama3_content_r1.json" --test ../data/raw/test_spark.txt --output "../data/rep/dbrd_test_spark_llama3_content_r1.txt"

python classical_approach/generate_input_dbrd.py --database "../data/ablation/spark_kpminer-idf_content_r1.json" --test ../data/raw/test_spark.txt --output "../data/rep/dbrd_test_spark_kpminer-idf_content_r1.txt"

python classical_approach/generate_input_dbrd.py --database "../data/ablation/spark_tfidf-idf_content_r1.json" --test ../data/raw/test_spark.txt --output "../data/rep/dbrd_test_spark_tfidf-idf_content_r1.txt"

# python classical_approach/generate_input_dbrd.py --database "../data/ablation/spark_yake_content_r1.json" --test ../data/raw/test_spark.txt --output "../data/rep/dbrd_test_spark_yake_content_r1.txt"

