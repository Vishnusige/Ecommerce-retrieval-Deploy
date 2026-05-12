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



def main():
    # 1. Load the stable model
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = model.to(device)
    model.eval()
    # Professional Pathing: Get the directory where app.py is located

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(curr_dir, "Data", "data", "flipkart_com-ecommerce_sample.csv")
    df = pd.read_csv(data_path)
    # Apply the same robust pathing to your numpy file
    embeddings_path = os.path.join(curr_dir, "embeddings.npy")
    all_embeddings = np.load(embeddings_path).astype('float32')
    dimension = all_embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings)
    ids_path = os.path.join(curr_dir, "id_list.npy")   
    ids = np.load(ids_path, allow_pickle=True)
    st.title("Retrieval Search")
    user_text = st.text_area('Enter your query below', value="A red skirt")
    generate_response_btn = st.button('Search for products!')
    
    if generate_response_btn and user_text is not None:
        emb = preprocess(tokenizer, model, user_text)
        distances, idx = find_similar(emb)
        idx = np.array(idx).flatten()
        
        # --- DIAGNOSTIC INJECTION ---
        st.error(f"DEBUG: Are embeddings corrupted? {np.isnan(emb).any()}")
        st.error(f"DEBUG: What did FAISS return? {idx}")
        st.error(f"DEBUG: First 3 IDs in the file: {ids[:3]}")
        # ----------------------------
        
        uniq_ids = [ids[i] for i in idx]
        images_links, product_names = get_images_list(df, uniq_ids)

        # Display the results
        st.write("**Products**:")
        
        if not images_links:
            st.error("No valid images could be extracted for this query.")
            
        for image_list in images_links:
            st.write(product_names[images_links.index(image_list)])
            cols = st.columns(len(image_list), gap="medium")
            for i, image_link in enumerate(image_list):
                with cols[i]:
                    try:
                        response = requests.get(image_link)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content))
                        st.image(image)
                    except Exception as e:
                        st.error("Could not load image. Link might be broken.")

if __name__ == "__main__":
    main()