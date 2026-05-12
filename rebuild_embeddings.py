import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import os

# 1. Setup paths
curr_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(curr_dir, "Data", "data", "flipkart_com-ecommerce_sample.csv")

# 2. Load the stable model (384 dimensions)
print("Loading MiniLM model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Load your CSV
print("Reading CSV data...")
df = pd.read_csv(data_path)

# Fill missing descriptions to avoid errors
df['description'] = df['description'].fillna('')

# 4. Generate Embeddings
print(f"Generating embeddings for {len(df)} products. This may take a few minutes...")
# encode() automatically handles batching and normalization
embeddings = model.encode(df['description'].tolist(), show_progress_bar=True)

# 5. Save the new 384-dimension files
np.save('embeddings.npy', embeddings.astype('float32'))
np.save('id_list.npy', df['uniq_id'].astype(str).values)

print("✅ Success!")
print(f"New embeddings shape: {embeddings.shape}") 
print("Your embeddings.npy is now 384-dimensional and ready for GitHub.")