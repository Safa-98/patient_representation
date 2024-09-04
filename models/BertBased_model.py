import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

def save_bert_embeddings(input_csv, save_dir):
    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Load dataset
    data = pd.read_csv(input_csv)

    # Create directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    for note_id, note in zip(data['NOTE_ID'], data['TEXT']):
        # Tokenize the entire note
        encoded_note = tokenizer.encode_plus(
            note,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded_note['input_ids'][0]
        attention_mask = encoded_note['attention_mask'][0]

        # Send the tokenized note to the BERT model
        with torch.no_grad():
            outputs = bert_model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
        
        # Extract the embedding corresponding to the [CLS] token
        note_document_embedding = outputs[0][:, 0, :]

        #uncomment these lines to learn embedddings used TE representations
        # Extract the embedding (mean of all tokens) # TE Representation
        #note_document_embedding = torch.mean(outputs[0], dim=1)  

        # Save the tensor with Note_id in the file name
        torch.save(note_document_embedding, os.path.join(save_dir, f'{note_id}.pt'))

if __name__ == "__main__":
    save_bert_embeddings('data/raw/physician_dataset.csv', 'data/processed/embed_BERT_CLS_note/')
