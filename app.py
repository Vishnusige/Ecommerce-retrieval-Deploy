import ast
import os
import re
from io import BytesIO

import faiss
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "6"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "2.5"))
PLACEHOLDER_IMAGE = "https://placehold.co/320x240?text=No+Image"

device = torch.device("cpu")


def mean_pool(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def clean_query(text):
    text = text.lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model.to(device)


@st.cache_data(show_spinner="Loading product catalog and embeddings...")
def load_catalog_and_vectors():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.getenv(
        "PRODUCT_DATA_PATH",
        os.path.join(curr_dir, "Data", "data", "flipkart_com-ecommerce_sample.csv"),
    )
    embeddings_path = os.getenv("EMBEDDINGS_PATH", os.path.join(curr_dir, "embeddings.npy"))
    ids_path = os.getenv("ID_LIST_PATH", os.path.join(curr_dir, "id_list.npy"))

    missing = [path for path in (data_path, embeddings_path, ids_path) if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing deployment artifact(s): " + ", ".join(missing))

    df = pd.read_csv(data_path)
    embeddings = np.load(embeddings_path).astype("float32")
    ids = np.load(ids_path, allow_pickle=True).astype(str)

    if len(df) != len(ids) or embeddings.shape[0] != len(ids):
        raise ValueError(
            "Catalog, embeddings, and ID list are out of sync: "
            f"catalog={len(df)}, embeddings={embeddings.shape[0]}, ids={len(ids)}"
        )

    return df, embeddings, ids


@st.cache_resource(show_spinner="Preparing vector index...")
def build_index(_embeddings):
    index = faiss.IndexFlatL2(_embeddings.shape[1])
    index.add(_embeddings)
    return index


def embed_query(tokenizer, model, text):
    cleaned = clean_query(text)
    if not cleaned:
        return None

    encoded_input = tokenizer(
        cleaned,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)

    embedding = mean_pool(model_output, encoded_input["attention_mask"])
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding.detach().cpu().numpy().astype("float32")


def parse_images(value):
    if not isinstance(value, str) or not value.strip():
        return []

    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []

    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, str) and item.startswith("http")]
    return []


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_image(url):
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception:
        return None


def search_products(df, ids, index, query_embedding, top_k):
    distances, result_indices = index.search(query_embedding, k=top_k)
    matches = []

    for distance, result_index in zip(distances[0], result_indices[0]):
        product_id = str(ids[result_index])
        row = df.loc[df["uniq_id"].astype(str) == product_id]
        if row.empty:
            continue

        product = row.iloc[0]
        matches.append(
            {
                "name": product.get("product_name", "Unknown product"),
                "description": product.get("description", ""),
                "price": product.get("discounted_price", ""),
                "retail_price": product.get("retail_price", ""),
                "score": float(distance),
                "images": parse_images(product.get("image", "")),
            }
        )

    return matches


def render_result(match):
    st.subheader(match["name"])

    meta = []
    if pd.notna(match["price"]) and str(match["price"]).strip():
        meta.append(f"Price: Rs. {match['price']}")
    if pd.notna(match["retail_price"]) and str(match["retail_price"]).strip():
        meta.append(f"MRP: Rs. {match['retail_price']}")
    if meta:
        st.caption(" | ".join(meta))

    images = match["images"][:4] or [PLACEHOLDER_IMAGE]
    cols = st.columns(min(len(images), 4), gap="small")
    for col, image_url in zip(cols, images):
        with col:
            image = fetch_image(image_url)
            if image is not None:
                st.image(image, use_container_width=True)
            else:
                st.image(PLACEHOLDER_IMAGE, use_container_width=True)

    description = str(match["description"])
    if description and description.lower() != "nan":
        st.write(description[:450] + ("..." if len(description) > 450 else ""))


def main():
    st.set_page_config(page_title="E-commerce Semantic Product Search", layout="wide")
    st.title("E-commerce Semantic Product Search")
    st.caption("Vector search over Flipkart product metadata using MiniLM embeddings and FAISS.")

    try:
        df, embeddings, ids = load_catalog_and_vectors()
        index = build_index(embeddings)
        tokenizer, model = load_model()
    except Exception as exc:
        st.error("The app could not load its deployment artifacts.")
        st.exception(exc)
        st.stop()

    with st.sidebar:
        st.metric("Products indexed", f"{len(ids):,}")
        st.metric("Embedding dimensions", embeddings.shape[1])
        st.write("Model:", MODEL_NAME)

    user_text = st.text_input("Search products", value="red skirt", placeholder="Try: running shoes, cotton saree")
    search_clicked = st.button("Search", type="primary", use_container_width=False)

    if search_clicked:
        query_embedding = embed_query(tokenizer, model, user_text)
        if query_embedding is None:
            st.warning("Enter a search query to retrieve products.")
            st.stop()

        matches = search_products(df, ids, index, query_embedding, TOP_K)
        if not matches:
            st.warning("No matching products found.")
            st.stop()

        st.write(f"Showing top {len(matches)} semantic matches")
        for match in matches:
            with st.container(border=True):
                render_result(match)


if __name__ == "__main__":
    main()
