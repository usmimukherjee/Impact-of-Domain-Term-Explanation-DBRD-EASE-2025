import pandas as pd
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer, util
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import networkx as nx
import numpy as np
import gc
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm 
from scipy.spatial.distance import cdist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model_deberta = AutoModelForSequenceClassification.from_pretrained("./dberta_finetuned").to(device)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalpha()]
    return ' '.join(clean_tokens)

def train_word2vec(sentences, vector_size=100, window=5, min_count=1):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return model

def create_similarity_graph(model):
    words = list(model.wv.index_to_key)
    vectors = np.array([model.wv[word] for word in words])  
    sim_matrix = 1 - squareform(pdist(vectors, metric='cosine'))  

    graph = nx.Graph()
    for i, word in enumerate(words):
        for j in range(i + 1, len(words)): 
            if sim_matrix[i, j] > 0.5:
                graph.add_edge(word, words[j], weight=sim_matrix[i, j])
    return graph


def calculate_pagerank(graph):
    pagerank_scores = nx.pagerank(graph, weight='weight')
    return sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

def calculate_sentence_similarity(sentence1, sentence2):
    embeddings1 = sbert_model.encode(sentences1)
    embeddings2 = sbert_model.encode(sentences2)
    cosine_sim = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_sim

def classify_reports_batched(reports, batch_size=8):
    preds = []
    for i in tqdm(range(0, len(reports), batch_size), desc="Classifying Reports"):
        torch.cuda.empty_cache()
        gc.collect()
        batch = reports[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = trained_model_deberta(**inputs)
        preds.extend(outputs.logits.argmax(-1).cpu().numpy())
    return np.array(preds)

def ensemble_learning(word2vec_scores, sbert_scores, deberta_preds):
    word2vec_mean = word2vec_scores.mean(axis=1)
    sbert_mean = sbert_scores.mean(axis=1)
    final_scores = (word2vec_mean + sbert_mean + deberta_preds) / 3
    decisions = final_scores > 0.5
    return decisions


def sentence_embeddings_from_word2vec(model, sentences1, sentences2):
    def compute_sentence_vector(sentence):
        vectors = [model.wv[word] for word in sentence if word in model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(model.vector_size)
    embeddings1 = np.array([compute_sentence_vector(sentence) for sentence in sentences1])
    embeddings2 = np.array([compute_sentence_vector(sentence) for sentence in sentences2])
    pairwise_similarity = 1 - cdist(embeddings1, embeddings2, metric='cosine')

    return pairwise_similarity

bug_reports = pd.read_csv('')
bug_reports['processed_text1'] = bug_reports['description1'].apply(preprocess_text)
bug_reports['processed_text2'] = bug_reports['description2'].apply(preprocess_text)


sentences1 = [text.split() for text in bug_reports['processed_text1']]
sentences2 = [text.split() for text in bug_reports['processed_text2']]
word2vec_model = train_word2vec(sentences1 + sentences2)  

similarity_graph = create_similarity_graph(word2vec_model)
pagerank_scores = calculate_pagerank(similarity_graph)
top_terms = [term for term, _ in pagerank_scores[:10]]  


sbert_similarity = calculate_sentence_similarity(
    bug_reports['processed_text1'].tolist(),
    bug_reports['processed_text2'].tolist()
)




deberta_preds = classify_reports_batched(
    bug_reports[['description1', 'description2']].values.tolist()
)

word2vec_scores = sentence_embeddings_from_word2vec(word2vec_model, sentences1, sentences2)
sbert_scores = sbert_similarity.numpy()

final_decisions = ensemble_learning(word2vec_scores, sbert_scores, deberta_preds)

bug_reports['is_similar_pred'] = final_decisions
print("Final Decisions:", bug_reports[['is_similar', 'is_similar_pred']])


labels = bug_reports['is_similar'].values
precision = precision_score(labels, final_decisions)
recall = recall_score(labels, final_decisions)
f1 = f1_score(labels, final_decisions)
roc_auc = roc_auc_score(labels, final_decisions)

print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC Score: {roc_auc}")
