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
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
model_deberta = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=2).to(device)


class BugReportDataset(Dataset):
    def __init__(self, descriptions1, descriptions2, labels, tokenizer, max_length=512):
        self.descriptions1 = descriptions1
        self.descriptions2 = descriptions2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
        self.descriptions1[idx],
        self.descriptions2[idx],
        truncation=True,
        padding="max_length",
        max_length=self.max_length,
        return_overflowing_tokens=False,
        return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Training function for DeBERTa
def train_deberta(model, train_loader, optimizer, device, num_epochs=10):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader):.4f}")


# Load dataset
bug_reports = pd.read_csv('')

# Preprocess text
bug_reports['processed_text1'] = bug_reports['description1']
bug_reports['processed_text2'] = bug_reports['description2']

# Prepare data for DeBERTa training
train_texts1 = bug_reports['processed_text1'].tolist()
train_texts2 = bug_reports['processed_text2'].tolist()
train_labels = bug_reports['is_similar'].tolist()

# Create Dataset and DataLoader
train_dataset = BugReportDataset(train_texts1, train_texts2, train_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_deberta.to(device)
optimizer = AdamW(model_deberta.parameters(), lr=0.0001)

# Train the DeBERTa model
train_deberta(model_deberta, train_loader, optimizer, device, num_epochs=10)

# Save the fine-tuned model
model_deberta.save_pretrained('./deberta_finetuned')
tokenizer.save_pretrained('./deberta_finetuned')