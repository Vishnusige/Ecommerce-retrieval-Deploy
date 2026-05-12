import os
import re
import torch
import requests
import numpy as np
import pandas as pd
import faiss
from PIL import Image
from io import BytesIO
import streamlit as st
import nltk
from nltk.corpus import stopwords
from similarity import find_similar
from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(tokenizer, model, text):
    # This model is stable and doesn't need position_id hacks
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Standard Mean Pooling (Better for this specific model)
    token_embeddings = model_output[0]
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    import torch.nn.functional as F
    emb = F.normalize(emb, p=2, dim=1)
    return emb.detach().cpu().numpy()

def preprocess(tokenizer, model, text):
    nltk.download('stopwords')
    text = text.lower()

    text = re.sub('<[^>]+>', '', text)
    
    text = re.sub(r'[^\w\s]', '', text)
    
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    emb = get_embeddings(tokenizer, model, text)
    return emb



def get_images_list(df, uniq_ids):
    images_list = []
    product_names = []
    
    for id in uniq_ids:
        matched_row = df[df['uniq_id'].astype(str) == str(id)]
        
        if not matched_row.empty:
            try:
                img_str = matched_row['image'].values[0]
                name_val = matched_row['product_name'].values[0]
                img_list = eval(img_str)
                images_list.append(img_list)
                product_names.append(name_val)
            except Exception as e:
                continue
        else:
            # If a mismatch happens, display it on the screen so we aren't flying blind
            st.warning(f"Data Mismatch: FAISS returned ID {id}, but it is missing from the CSV.")
            
    return images_list, product_names



import faiss  # Ensure this is lowercase at the top of your script

import faiss # Ensure lowercase 'f'

def main():
    # 1. Load the stable model (384 dimensions)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = model.to(device)
    model.eval()

    # 2. Setup paths
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(curr_dir, "Data", "data", "flipkart_com-ecommerce_sample.csv")
    df = pd.read_csv(data_path)
    
    embeddings_path = os.path.join(curr_dir, "embeddings.npy")
    ids_path = os.path.join(curr_dir, "id_list.npy")
    
    # 3. Load data
    # We load the embeddings you already have and force them to float32
    all_embeddings = np.load(embeddings_path).astype('float32')
    ids = np.load(ids_path, allow_pickle=True)

    # 4. THE FIX: Build a fresh index in memory
    # This ignores any old files and creates an index that matches your data EXACTLY
    dimension = all_embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings) 
    
    st.title("Retrieval Search")
    user_text = st.text_area('Enter your query below', value="A red skirt")
    generate_response_btn = st.button('Search for products!')
    
    if generate_response_btn and user_text:
        # Preprocess text and get query embedding (384 dims)
        emb = preprocess(tokenizer, model, user_text)
        
        # 5. Search the memory-resident index
        # This ensures the dimensions (384 vs 384) always match
        distances, idx = index.search(emb.reshape(1, -1), k=6)
        idx = np.array(idx).flatten()
        
        uniq_ids = [ids[i] for i in idx]
        images_links, product_names = get_images_list(df, uniq_ids)

        st.write("**Products**:")
        if not images_links:
            st.error("No products found.")
            
        for i in range(len(images_links)):
            st.write(product_names[i])
            cols = st.columns(len(images_links[i]), gap="medium")
            for j, link in enumerate(images_links[i]):
                with cols[j]:
                    try:
                        response = requests.get(link)
                        st.image(Image.open(BytesIO(response.content)))
                    except:
                        continue

if __name__ == "__main__":
    main()