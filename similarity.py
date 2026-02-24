import os
import faiss
import numpy as np

def build_and_save_faiss_index(embeddings, index_path):
    d = embeddings.shape[1]
    # print(d)
    index = faiss.IndexFlatL2(d)  # L2 distance
    index.add(embeddings)
    
    faiss.write_index(index, index_path)
    print(f"Index saved to {index_path}")
    return index

def load_faiss_index(index_path):
    print(f"Attempting to load index from {index_path}")
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print(f"Index loaded from {index_path}")
        print(f"Index type: {type(index)}")
        return index
    else:
        print(f"No index found at {index_path}. Building Index..")
        embeddings = np.load("embeddings.npy")
        print(f"Embeddings shape: {embeddings.shape}")
        index = build_and_save_faiss_index(embeddings, index_path)
        print(f"Built index type: {type(index)}")
        return index


def find_similar(query_embedding, k = 6):
    index = load_faiss_index("index")
    distances, indices = index.search(query_embedding.reshape(1, -1), k) # (1, 768)
    return distances[0], indices[0]