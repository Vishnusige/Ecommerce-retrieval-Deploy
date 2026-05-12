import os
import re
import torch
import requests
import numpy as np
import pandas as pd
import faiss  # Corrected lowercase import
from PIL import Image
from io import BytesIO
import streamlit as st
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel

# Set device to CPU for Streamlit Cloud stability
device = torch.device('cpu')

def get_embeddings(tokenizer, model, text):
    """Generates 384-dimensional embeddings using the stable MiniLM model."""
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Mean Pooling
    token_embeddings = model_output[0]
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # L2 Normalization is critical for accurate FAISS L2 distance searching
    import torch.nn.functional as F
    emb = F.normalize(emb, p=2, dim=1)
    return emb.detach().cpu().numpy()

def preprocess(tokenizer, model, text):
    """Cleans user input and returns the vector embedding."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        
    text = text.lower()
    text = re.sub('<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return get_embeddings(tokenizer, model, text)

def get_images_list(df, uniq_ids):
    """Fetches product names and image lists while handling missing IDs."""
    images_list = []
    product_names = []
    
    for id in uniq_ids:
        matched_row = df[df['uniq_id'].astype(str) == str(id)]
        if not matched_row.empty:
            try:
                img_str = matched_row['image'].values[0]
                name_val = matched_row['product_name'].values[0]
                images_list.append(eval(img_str))
                product_names.append(name_val)
            except:
                continue
    return images_list, product_names

def main():
    st.title("Retrieval Search")
    
    # 1. Load Stable Model (MiniLM-L6-v2)
    @st.cache_resource
    def load_model():
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        return tokenizer, model.to(device)

    tokenizer, model = load_model()

    # 2. Define Paths and Load Data
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(curr_dir, "Data", "data", "flipkart_com-ecommerce_sample.csv")
    embeddings_path = os.path.join(curr_dir, "embeddings.npy")
    ids_path = os.path.join(curr_dir, "id_list.npy")

    df = pd.read_csv(data_path)
    all_embeddings = np.load(embeddings_path).astype('float32')
    ids = np.load(ids_path, allow_pickle=True)

    # 3. Dynamic Indexing (The Loop Breaker)
    # This rebuilds the index to match the embeddings.npy file dimensions exactly
    dimension = all_embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings)

    # 4. User Interface
    user_text = st.text_area('Enter your query below', value="A red skirt")
    generate_response_btn = st.button('Search for products!')
    
    if generate_response_btn and user_text:
        emb = preprocess(tokenizer, model, user_text)
        
        # Search against the memory-resident index
        distances, idx = index.search(emb.reshape(1, -1), k=6)
        idx = np.array(idx).flatten()
        
        uniq_ids = [ids[i] for i in idx]
        images_links, product_names = get_images_list(df, uniq_ids)

        st.write("**Results:**")
        if not images_links:
            st.error("No matches found in the database.")
            
        for i in range(len(images_links)):
            st.subheader(product_names[i])
            cols = st.columns(len(images_links[i]), gap="small")
            for j, link in enumerate(images_links[i]):
                with cols[j]:
                    try:
                        st.image(link, use_container_width=True)
                    except:
                        continue

if __name__ == "__main__":
    main()