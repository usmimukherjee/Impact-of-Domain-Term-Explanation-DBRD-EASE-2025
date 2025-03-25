

# !pip install transformers

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import networkx as nx
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import string
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


import csv
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string
import networkx as nx
import nltk
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Reading the input CSV
input_file = '/'  # Path to your input CSV file
output_file = ''  # Path to your output CSV file

data = pd.read_csv(input_file)
results = []

for index, row in data.iterrows():
    bug_id = row['Bug ID']
    text = row['Description']

    stop_words = set(stopwords.words('english'))
    special_chars = set(''':;"'.,><`~()_-+=[]{}/\\|*?!@$%^#&''')

    def filter_phrase(phrase):
        words = phrase.lower().split()

        if any(word in stop_words for word in words):
            return False
        if any(char.isdigit() for char in phrase):
            return False

        if any(char in special_chars for char in phrase):
            return False

        return True

    tokenizer = AutoTokenizer.from_pretrained("jeniya/BERTOverflow")
    model = AutoModelForTokenClassification.from_pretrained("jeniya/BERTOverflow")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    entities = nlp(text)

    filtered_entities = []

    text2 = ""

    for entity in entities:
        word = entity['word']
        score = entity['score']
        text2 += word + " "

        if filter_phrase(word):
            filtered_entities.append((word, score))

    SERvalues={}

    SERvalues={}
    avg = 0
    for word, score in filtered_entities:
        SERvalues[word] = score
        avg += score
    if(len(SERvalues)==0):
        continue
    avg /= len(SERvalues)

    # print(SERvalues)


    # Jespersen Rank Theory
    class POSRankProvider:
        def __init__(self, graph, tokendb, initial_vertex_score=1.0, damping_factor=0.85, max_iteration=100, significance_threshold=0.0001, weight_bias=0.1):
            self.graph = graph
            self.tokendb = tokendb
            self.initial_vertex_score = initial_vertex_score
            self.damping_factor = damping_factor
            self.max_iteration = max_iteration
            self.significance_threshold = significance_threshold
            self.weight_bias = weight_bias
            self.old_score_map = {node: self.initial_vertex_score for node in self.graph.nodes}
            self.new_score_map = {node: self.initial_vertex_score for node in self.graph.nodes}

        def check_significant_diff(self, old_v, new_v):
            diff = abs(new_v - old_v)
            return diff > self.significance_threshold

        def calculate_pos_rank(self):
            d = self.damping_factor
            N = len(self.graph.nodes)
            itercount = 0
            enough_iteration = False

            while not enough_iteration and itercount < self.max_iteration:
                insignificant = 0
                for node in self.graph.nodes:
                    incoming_edges = self.graph.in_edges(node, data=True)
                    trank = 1.0 - d
                    coming_score = 0.0

                    for edge in incoming_edges:
                        source = edge[0]
                        outdegree = len(self.graph.out_edges(source))
                        score = self.old_score_map[source]
                        edge_weight = edge[2].get('weight', 1.0)

                        # Adjust weight_bias dynamically
                        source_pos = pos_tag([source])[0][1]
                        if source_pos.startswith('NN'):  # Increase bias for nouns
                            dynamic_weight_bias = self.weight_bias * 1.5
                        elif source_pos.startswith('JJ'):  # Decrease bias for adjectives
                            dynamic_weight_bias = self.weight_bias * 0.8
                        elif source_pos.startswith('VB'):  # Increase bias for verbs
                            dynamic_weight_bias = self.weight_bias * 1.2
                        elif source_pos.startswith('RB'):  # Decrease bias for adverbs
                            dynamic_weight_bias = self.weight_bias * 0.5

                        else:
                            dynamic_weight_bias = self.weight_bias

                        if outdegree == 0:
                            coming_score += score
                        else:
                            coming_score += (score / outdegree) * edge_weight * dynamic_weight_bias

                    coming_score *= d
                    trank += coming_score

                    if self.check_significant_diff(self.old_score_map[node], trank):
                        self.new_score_map[node] = trank
                    else:
                        insignificant += 1

                self.old_score_map = self.new_score_map.copy()
                itercount += 1

                if insignificant == N:
                    enough_iteration = True

            self.record_normalize_scores()
            return self.tokendb

        def record_normalize_scores(self):
            max_rank = max(self.new_score_map.values())
            for key in self.new_score_map:
                score = self.new_score_map[key] / max_rank
                self.tokendb[key].pos_rank_score = score

    class QueryToken:
        def __init__(self, term):
            self.term = term
            self.pos_rank_score = 0.0

        def __repr__(self):
            return f"QueryToken(term={self.term}, pos_rank_score={self.pos_rank_score})"



    # Tokenization and POS tagging
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    graph = nx.DiGraph()

    # Building the graph
    previous_word = None
    for word, pos in pos_tags:
        graph.add_node(word)
        if previous_word:
            graph.add_edge(previous_word, word, weight=1.0)
        previous_word = word

    # Initializing token database and POS rank provider
    tokendb = {word: QueryToken(word) for word in graph.nodes}
    pos_rank_provider = POSRankProvider(graph, tokendb, weight_bias=avg)  # Provide an appropriate weight_bias

    # Calculating POS rank
    tokendb = pos_rank_provider.calculate_pos_rank()

    # Inverting POS rank score for ranking purposes
    for token in tokendb.values():
        token.pos_rank_score = 1 / token.pos_rank_score

    # Sorting tokens based on POS rank score
    sorted_tokens = sorted(tokendb.values(), key=lambda token: token.pos_rank_score, reverse=True)

    # Post-processing: Filtering phrases
    stop_words = set(stopwords.words('english'))
    special_chars = set(string.punctuation)

    def filter_phrase(phrase):
        words = phrase.lower().split()

        if any(word in stop_words for word in words):
            return False
        if any(char.isdigit() for char in phrase):
            return False
        if any(char in special_chars for char in phrase):
            return False

        return True

    filtered_tokens_with_scores = {token.term: token.pos_rank_score for token in sorted(tokendb.values(), key=lambda token: token.pos_rank_score, reverse=True) if filter_phrase(token.term)}

    #normalizing scores and printing the words and their scores
    def print_normalized_scores(filtered_tokens_with_scores):
        total_score = sum(filtered_tokens_with_scores.values())
        normalized_scores = {token: score / total_score for token, score in filtered_tokens_with_scores.items()}
        # print("\nNormalized Scores:")
        # for token, score in normalized_scores.items():
        #     print(f"{token}: {score}")

        return normalized_scores

    POSvalues = print_normalized_scores(filtered_tokens_with_scores)

    # print(POSvalues)


    '''
    Combination Based Scoring
    '''


    # Combine the scores
    def combine_scores(pos_score, ner_score, alpha=0.5):
        return alpha * pos_score + (1 - alpha) * ner_score

    combined_scores = {}
    for word in POSvalues:
        pos_score = POSvalues.get(word, 0)
        ner_score = SERvalues.get(word, 0)
        combined_scores[word] = combine_scores(pos_score, ner_score)

    # Generate Candidate Bigrams
    tokenized_document = word_tokenize(text2.lower())

    candidate_bigrams = [(tokenized_document[i], tokenized_document[i + 1]) for i in range(len(tokenized_document) - 1)]

    # Score Bigrams
    bigram_scores = Counter()
    for bigram in candidate_bigrams:
        word1, word2 = bigram
        if word1 in combined_scores and word2 in combined_scores:
            bigram_scores[bigram] = (combined_scores[word1] + combined_scores[word2]) / 2

    # Filter and Select Top Bigrams
    stop_words = set(stopwords.words('english'))
    special_chars = set(string.punctuation)
    top_n = 5

    def is_valid_bigram(bigram):
        word1, word2 = bigram
        if any(word in stop_words for word in bigram):
            return False
        if any(char.isdigit() for char in word1 + word2):
            return False
        if any(char in special_chars for char in word1 + word2):
            return False
        return True

    filtered_bigrams = {bigram: score for bigram, score in bigram_scores.items() if is_valid_bigram(bigram)}

    # Sort and print top bigrams
    top_bigrams = sorted(filtered_bigrams.items(), key=lambda item: item[1], reverse=True)[:top_n]

    # Preparing the output as a list of bigrams
    bigram_list = [' '.join(bigram) for bigram, score in top_bigrams]
    results.append({'Bug ID': bug_id, 'Bigrams': bigram_list})

# Writing the results to output CSV
output_df = pd.DataFrame(results)
output_df.to_csv(output_file, index=False)

print(filtered_entities)

