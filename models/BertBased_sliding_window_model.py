import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# Constants
MAX_LENGTH = 512  # Maximum sequence length allowed by BERT
STRIDE = 256      # Stride size for the sliding window

# Directory to save individual tensors
SAVE_DIR = 'data/processed/embed_BERT_sliding_note/'
os.makedirs(SAVE_DIR, exist_ok=True)

def load_model():
    """Load the BERT tokenizer and model."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, bert_model


def process_sentence(sentence, tokenizer, bert_model):
    """Process a sentence to obtain BERT embeddings."""
    encoded_sentence = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded_sentence['input_ids'][0]
    attention_mask = encoded_sentence['attention_mask'][0]
    with torch.no_grad():
        outputs = bert_model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
    # Use the [CLS] token representation
    embeddings = outputs[0][:, 0, :]

    #uncomment these lines to learn embedddings used TE representations
    # Extract the embedding (mean of all tokens) # TE Representation
    #embeddings = torch.mean(outputs[0], dim=1) 

    return embeddings

def process_dataset(input_csv, tokenizer, bert_model):
    """Process a dataset to extract and save BERT embeddings."""
    data = pd.read_csv(input_csv)
    
    for note_id, note in zip(data['NOTE_ID'], data['TEXT']):
        sentences = note.split('. ')
        note_embeddings_per_sentence = []
        
        for sentence in sentences:
            if len(sentence) > MAX_LENGTH:
                # Sliding window approach for long sentences
                start = 0
                while start < len(sentence):
                    end = start + MAX_LENGTH
                    if end > len(sentence):
                        end = len(sentence)
                    note_embeddings_per_sentence.append(process_sentence(sentence[start:end], tokenizer, bert_model))
                    start += STRIDE
            else:
                note_embeddings_per_sentence.append(process_sentence(sentence, tokenizer, bert_model))
        
        # Aggregate sentence embeddings to obtain document-level embeddings
        note_document_embedding = torch.mean(torch.stack(note_embeddings_per_sentence), dim=0)

        # Save individual tensors with Note_id in the file name
        torch.save(note_document_embedding, os.path.join(SAVE_DIR, f'{note_id}.pt'))

if __name__ == "__main__":
    tokenizer, bert_model = load_model()
    
    # Path to the dataset CSV file
    input_csv = 'data/raw/physician_dataset.csv'
    
    # Process the dataset
    process_dataset(input_csv, tokenizer, bert_model)
