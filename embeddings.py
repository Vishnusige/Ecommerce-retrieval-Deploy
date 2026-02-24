import torch
import faiss
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

df = pd.read_csv('preprocessed_text.csv')

tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-base-en-v1.5")
model = AutoModel.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0] #  Shape: (batch_size, hidden_size)

def get_embeddings(text):
    encoded_input = tokenizer(
        text, padding=True, truncation=True, return_tensors="pt"
    )
    # Shape of encoded_input: 
    #   {'input_ids': (batch_size, sequence_length),
    #    'attention_mask': (batch_size, sequence_length)}
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Shape of model_output.last_hidden_state: (batch_size, sequence_length, hidden_size)

    # Pool the output and move to CPU
    return cls_pooling(model_output).detach().cpu().numpy() 

batch_size = 32  # Adjust batch size based on GPU memory

# Initialize lists for embeddings and ID mappings
all_embeddings = []
id_list = []

# Generate embeddings in batches
for i in tqdm(range(0, len(df), batch_size)):
    end_idx = min(i + batch_size, len(df))
    texts_batch = df['text_col'].iloc[i:end_idx].tolist()
    ids_batch = df['uniq_id'].iloc[i:end_idx].tolist()
    
    if not texts_batch:  # Skip empty batches
        continue
    
    # Ensure that IDs and texts are of the same length
    if len(texts_batch) != len(ids_batch):
        raise ValueError("Mismatch between number of texts and IDs in the batch.")
    
    embeddings_batch = get_embeddings(texts_batch)
    all_embeddings.append(embeddings_batch)
    id_list.extend(ids_batch)

# Convert list of embeddings to a numpy array
embeddings = np.vstack(all_embeddings)

# Save embeddings to a file
np.save('embeddings.npy', embeddings)

# Save ID list to a file
np.save('id_list.npy', np.array(id_list))