import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import pandas as pd

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadAttentionWrapper, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, encoder_outputs):
        attn_output, attn_weights = self.multihead_attn(encoder_outputs, encoder_outputs, encoder_outputs)
        return attn_output, attn_weights

class BiRNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BiRNNEncoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, x):
        outputs, _ = self.rnn(x)
        return outputs

class HierarchicalAttentionModel(nn.Module):
    def __init__(self, bert_model, hidden_dim, num_heads):
        super(HierarchicalAttentionModel, self).__init__()
        self.bert_model = bert_model
        self.hidden_dim = hidden_dim

        self.word_encoder = BiRNNEncoder(bert_model.config.hidden_size, hidden_dim)
        self.word_attention = MultiHeadAttentionWrapper(hidden_dim * 2, num_heads)
        self.sentence_encoder = BiRNNEncoder(hidden_dim * 2, hidden_dim)
        self.sentence_attention = MultiHeadAttentionWrapper(hidden_dim * 2, num_heads)
        self.document_encoder = BiRNNEncoder(hidden_dim * 2, hidden_dim)
        self.document_attention = MultiHeadAttentionWrapper(hidden_dim * 2, num_heads)

    def forward(self, input_ids, attention_mask):
        sentence_vectors = []

        for i in range(input_ids.size(1)):
            input_ids_single = input_ids[:, i, :]
            attention_mask_single = attention_mask[:, i, :]

            outputs = self.bert_model(input_ids_single, attention_mask=attention_mask_single)
            word_embeddings = outputs.last_hidden_state

            word_encoder_outputs = self.word_encoder(word_embeddings)
            sentence_vector, _ = self.word_attention(word_encoder_outputs)
            sentence_vector = sentence_vector.mean(dim=1)
            sentence_vectors.append(sentence_vector.unsqueeze(1))

        sentence_vectors = torch.cat(sentence_vectors, dim=1)
        sentence_encoder_outputs = self.sentence_encoder(sentence_vectors)
        document_vector, _ = self.sentence_attention(sentence_encoder_outputs)
        document_vector = torch.max(document_vector, dim=1)[0]

        return document_vector

def load_model_and_tokenizer():
    """Load BERT model and tokenizer."""
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return bert_model, tokenizer

def prepare_document(document, tokenizer, max_sentence_length, max_document_length):
    """Tokenize and encode the document into tensors."""
    sentences = [sent.strip() + '.' for sent in document.split('.') if sent.strip()]
    encoded_document = []

    for sentence in sentences[:max_document_length]:
        encoding = tokenizer(sentence, return_tensors="pt", padding='max_length', max_length=max_sentence_length, truncation=True)
        encoded_document.append({
            'input_ids': encoding["input_ids"].squeeze(0),
            'attention_mask': encoding["attention_mask"].squeeze(0)
        })

    if len(encoded_document) < max_document_length:
        padding_encoding = {
            'input_ids': torch.zeros(max_sentence_length, dtype=torch.long),
            'attention_mask': torch.zeros(max_sentence_length, dtype=torch.long)
        }
        while len(encoded_document) < max_document_length:
            encoded_document.append(padding_encoding)

    input_ids_tensor = torch.stack([sent['input_ids'] for sent in encoded_document])
    attention_mask_tensor = torch.stack([sent['attention_mask'] for sent in encoded_document])

    return input_ids_tensor.unsqueeze(0), attention_mask_tensor.unsqueeze(0)

def process_dataset(input_csv, model, tokenizer, output_dir, max_sentence_length=512, max_document_length=34):
    """Process a dataset to extract and save embeddings."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    documents = df['TEXT'].tolist()
    note_ids = df['NOTE_ID'].tolist()

    for idx, document in enumerate(documents):
        input_ids_tensor, attention_mask_tensor = prepare_document(
            document, tokenizer, max_sentence_length, max_document_length
        )
        
        with torch.no_grad():
            document_embedding = model(input_ids_tensor, attention_mask_tensor)

        note_id = note_ids[idx]
        embedding = document_embedding.squeeze(0).cpu()
        filepath = os.path.join(output_dir, f"{note_id}.pt")
        torch.save(embedding, filepath)
        print(f"Saved {filepath}")

if __name__ == "__main__":
    bert_model, tokenizer = load_model_and_tokenizer()
    hidden_dim = 256
    num_heads = 8
    model = HierarchicalAttentionModel(bert_model, hidden_dim, num_heads)
    model.eval()
    
    input_csv = "data/raw/physician_dataset.csv"
    output_dir = "data/processed/embed_HAN_multi_note/"
    
    process_dataset(input_csv, model, tokenizer, output_dir)
