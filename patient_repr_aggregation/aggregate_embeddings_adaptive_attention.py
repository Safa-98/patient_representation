import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the attention and model classes
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs):
        attention_weights = F.softmax(self.attention(encoder_outputs), dim=1)
        context_vector = torch.sum(encoder_outputs * attention_weights, dim=1)
        return context_vector, attention_weights

class BiRNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BiRNNEncoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
    
    def forward(self, x):
        outputs, _ = self.rnn(x)
        return outputs

class MultiNoteRepresentationModel(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(MultiNoteRepresentationModel, self).__init__()
        self.note_encoder = BiRNNEncoder(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.note_attention = Attention(hidden_dim * 2)
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Pooling to produce a fixed-size output
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Fully connected layer to produce the final representation
    
    def forward(self, note_embeddings_list):
        note_representations = []
        
        # Encode and apply attention to each note embedding
        for note_embedding in note_embeddings_list:
            encoded_note = self.note_encoder(note_embedding.unsqueeze(0))
            note_representation, _ = self.note_attention(encoded_note)
            note_representations.append(note_representation)
        
        # Stack all note representations
        stacked_representations = torch.stack(note_representations, dim=1)  # Shape: [batch_size=1, num_notes, hidden_dim * 2]
        
        # Apply pooling to handle variable number of notes
        pooled_representations = self.pooling(stacked_representations.permute(0, 2, 1)).squeeze(-1)  # Shape: [batch_size=1, hidden_dim * 2]
        
        # Debug: Print tensor shapes
        #print(f"Pooled representations shape: {pooled_representations.shape}")
        
        # Pass through fully connected layer to create final note representation
        final_note_representation = self.fc(pooled_representations)
        
        return final_note_representation

# Initialize the model
hidden_dim = 512  # Dimensionality of LSTM hidden states
output_dim = 256  # Size of the final representation
multi_note_model = MultiNoteRepresentationModel(hidden_dim, output_dim)
multi_note_model.eval()

# Load DataFrame
df = pd.read_csv("physician_patient_notes_excluding_1.csv")


# Group note embeddings by patient (SUBJECT_ID)
grouped = df.groupby('SUBJECT_ID')['NOTE_ID'].apply(list).to_dict()

tensor_dir = 'data/processed/embed_HAN_single_note/tensor_HAN_single_adapt_patient/'
os.makedirs(output_dir, exist_ok=True)

# Process each set of note_ids
for subject_id, note_ids in grouped.items():
    note_embeddings = []
    
    # Load all note embeddings for the current set of note_ids
    for note_id in note_ids:
        filepath = os.path.join("data/processed/embed_HAN_single_note", f"{note_id}.pt")
        if os.path.exists(filepath):
            note_embedding = torch.load(filepath)
            note_embeddings.append(note_embedding)
    
    if note_embeddings:
        # Move embeddings to the same device as the model
        note_embeddings = [embedding.to(next(multi_note_model.parameters()).device) for embedding in note_embeddings]

        # Forward pass to obtain the combined representation
        with torch.no_grad():
            combined_representation = multi_note_model(note_embeddings)

        # Save the combined representation
        combined_filepath = os.path.join(output_dir, f"{subject_id}.pt")
        torch.save(combined_representation.cpu(), combined_filepath)
        print(f"Saved {combined_filepath}")


