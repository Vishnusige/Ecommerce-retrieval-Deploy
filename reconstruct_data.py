import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords

# 1. Setup paths
curr_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_path = os.path.join(curr_dir, "Data", "data", "flipkart_com-ecommerce_sample.csv")

print("Loading raw data...")
df = pd.read_csv(raw_data_path)

# 2. Preprocessing Logic (Matches your app.py logic)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub('<[^>]+>', '', text) # Remove HTML
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

print("Cleaning text (this might take a minute)...")
# We use 'description' because that's usually the 'text_col' in this dataset
df['text_col'] = df['description'].apply(clean_text)

# 3. Save the missing file
output_path = os.path.join(curr_dir, 'preprocessed_text.csv')
# We only need the ID and the cleaned text for the embeddings script
df[['uniq_id', 'text_col']].to_csv(output_path, index=False)

print(f"Success! {output_path} has been created.")