import os
import torch
import faiss
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

# 1. Initialization and Configuration
df = pd.read_csv('preprocessed_text.csv')

tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-base-en-v1.5")
model = AutoModel.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing on: {device}")
model = model.to(device)

batch_size = 16  # Optimized for system RAM stability

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text):
    if not text or (isinstance(text, list) and not any(text)):
        return np.zeros((len(text) if isinstance(text, list) else 1, 768))

    encoded_input = tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        max_length=500, 
        return_tensors="pt"
    )
    
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # Explicit Position ID mapping to prevent boundary indexing errors
    seq_length = encoded_input['input_ids'].shape[1]
    position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(encoded_input['input_ids'].shape[0], -1)
    encoded_input['position_ids'] = position_ids

    with torch.no_grad():
        model_output = model(**encoded_input)

    return cls_pooling(model_output).detach().cpu().numpy() 

# 2. Resiliency and State Management
all_embeddings = []
id_list = []
start_batch = 0

if os.path.exists('embeddings.npy') and os.path.exists('id_list.npy'):
    try:
        existing_embeddings = np.load('embeddings.npy')
        existing_ids = np.load('id_list.npy')
        all_embeddings = [existing_embeddings]
        id_list = list(existing_ids)
        start_row = len(id_list)
        start_batch = start_row // batch_size
        print(f"Recovered secured state. Resuming from batch {start_batch} (Row {start_row})...")
    except Exception as e:
        print("Warning: Existing files are corrupted. Starting from batch 0.")
        all_embeddings = []
        id_list = []
        start_batch = 0
else:
    print("Initiating fresh data processing pipeline...")

# 3. Batch Processing with Checkpointing
for i in tqdm(range(start_batch * batch_size, len(df), batch_size)):
    end_idx = min(i + batch_size, len(df))
    texts_batch = df['text_col'].iloc[i:end_idx].fillna("").tolist()
    ids_batch = df['uniq_id'].iloc[i:end_idx].tolist()
    
    if not texts_batch:
        continue
    
    embeddings_batch = get_embeddings(texts_batch)
    all_embeddings.append(embeddings_batch)
    id_list.extend(ids_batch)
    
    # Auto-save checkpoint every 50 batches
    if (i // batch_size) % 50 == 0 and i > 0:
        np.save('embeddings.npy', np.vstack(all_embeddings))
        np.save('id_list.npy', np.array(id_list))

# 4. Final Aggregation and FAISS Index Compilation
print("Processing complete. Compiling final matrices...")
final_embeddings = np.vstack(all_embeddings)
np.save('embeddings.npy', final_embeddings)
np.save('id_list.npy', np.array(id_list))

print("Constructing FAISS structural index...")
d = final_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(final_embeddings)
faiss.write_index(index, "index")

print("System Ready: All pipeline artifacts successfully generated and secured.")