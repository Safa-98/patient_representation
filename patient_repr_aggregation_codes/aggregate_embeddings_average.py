import os
import pandas as pd
import torch

# Directory where note embeddings are stored
tensor_dir = 'data/processed/embed_BERT_CLS_note/tensor_BERT_CLS_avg_patient/'
os.makedirs(tensor_dir, exist_ok=True)

# Load your DataFrame
df = pd.read_csv("physician_patient_notes_excluding_1.csv")


# Aggregate note embeddings by averaging
for half_note_id in df['NOTE_ID'].str.split('_').str[0].unique():
    matching_rows = df[df['NOTE_ID'].str.startswith(half_note_id)]
    matching_tensors = [torch.load(os.path.join('data/processed/embed_BERT_CLS_note/', f"{note_id}.pt")) for note_id in matching_rows['NOTE_ID']]
    aggregated_embedding = torch.stack(matching_tensors).mean(dim=0)
    torch.save(aggregated_embedding, os.path.join(tensor_dir, f'{half_note_id}.pt'))
    print(f"Saved {half_note_id}.pt")
