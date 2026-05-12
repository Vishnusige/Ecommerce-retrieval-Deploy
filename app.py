import os
import re
import torch
import requests
import numpy as np
import pandas as pd
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
    encoded_input = tokenizer(
        text, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # Explicit Position ID mapping to prevent RoPE boundary indexing errors
    seq_length = encoded_input['input_ids'].shape[1]
    position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(encoded_input['input_ids'].shape[0], -1)
    encoded_input['position_ids'] = position_ids

    with torch.no_grad():
        model_output = model(**encoded_input)
        
    # Pool the output and move to CPU
    return cls_pooling(model_output).detach().cpu().numpy()

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
        # 1. Force both IDs to strings to prevent any hidden type-mismatch errors
        matched_row = df[df['uniq_id'].astype(str) == str(id)]
        
        # 2. Only proceed if the product actually exists in the CSV
        if not matched_row.empty:
            try:
                # Extract values safely
                img_str = matched_row['image'].values[0]
                name_val = matched_row['product_name'].values[0]
                
                # Parse the stringified list into an actual Python list
                img_list = eval(img_str)
                
                # Append to our final lists
                images_list.append(img_list)
                product_names.append(name_val)
            except Exception as e:
                # If a product has a corrupted image link or eval() fails, 
                # we just skip it quietly rather than crashing the app!
                continue
                
    return images_list, product_names



def main():
    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-base-en-v1.5")
    model = AutoModel.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)
    model = model.to(device)

    # Professional Pathing: Get the directory where app.py is located
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path relative to app.py
    # IMPORTANT: Ensure the capitalization here matches your GitHub folders EXACTLY
    data_path = os.path.join(curr_dir, "Data", "data", "flipkart_com-ecommerce_sample.csv")
    df = pd.read_csv(data_path)

    # Apply the same robust pathing to your numpy file
    ids_path = os.path.join(curr_dir, "id_list.npy")
    ids = np.load(ids_path)
    
    st.title("Retrieval Search")
    user_text = st.text_area('Enter your query below', value="A red skirt")
    generate_response_btn = st.button('Search for products!')
    
    if generate_response_btn and user_text is not None:
        emb = preprocess(tokenizer, model, user_text)
        distances, idx = find_similar(emb)
        uniq_ids = [ids[i] for i in idx]
        images_links, product_names = get_images_list(df, uniq_ids)

        # Display the results
        st.write("**Products**:")
        for image_list in images_links:
            st.write(product_names[images_links.index(image_list)])
            cols = st.columns(len(image_list), gap="medium")
            for i, image_link in enumerate(image_list):
                with cols[i]:
                    try:
                        response = requests.get(image_link)
                        response.raise_for_status() # Check for broken links
                        image = Image.open(BytesIO(response.content))
                        st.image(image)
                    except Exception as e:
                        st.error(f"Could not load image. Link might be broken.")

if __name__ == "__main__":
    main()