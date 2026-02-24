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
        ls = df[df['uniq_id'] == id]['image'].values[0]
        ls = eval(ls)
        images_list.append(ls)
        ls = df[df['uniq_id'] == id]['product_name'].values[0]
        product_names.append(ls)
    return images_list, product_names



def main():
    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-base-en-v1.5")
    model = AutoModel.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)
    model = model.to(device)

    curr_dir = os.getcwd()
    data_path = os.path.join(curr_dir, "Data", "data", "flipkart_com-ecommerce_sample.csv")
    df = pd.read_csv(data_path)

    ids = np.load("id_list.npy")
    st.title("Retrieval Search")
    user_text = st.text_area('Enter you query below', value = "A red skirt")
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
                    response = requests.get(image_link)
                    image = Image.open(BytesIO(response.content))
                    # image = image.resize((500, 500))
                    st.image(image)
                    

if __name__ == "__main__":
    main()    