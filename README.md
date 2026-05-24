# E-commerce Semantic Product Search

Semantic product retrieval demo for e-commerce catalogs. The app searches product meaning, not just exact keywords, by embedding product descriptions and comparing user queries against a FAISS vector index.

## Live Demo

Deployment-ready Streamlit app:

- App entrypoint: `app.py`
- Recommended deploy target: Streamlit Community Cloud or Render
- Repository with deployment artifacts: `Vishnusige/Ecommerce-retrieval-Deploy`
- Source/development repository: `Vishnusige/ecommerce-retrieval-search`

Add the final public URL here after deployment:

```text
https://<your-deployed-streamlit-or-render-url>
```

## Features

- Semantic search over 20,002 Flipkart catalog products
- MiniLM query embeddings with 384-dimensional vectors
- FAISS nearest-neighbor search for fast retrieval
- Product metadata display with image fallback handling
- Cached model, catalog, embeddings, and index loading for stable Streamlit runtime
- Deployment configuration for Streamlit Cloud and Render

## Architecture

```text
User query
  -> clean query text
  -> MiniLM embedding model
  -> normalized 384-d query vector
  -> FAISS vector search over precomputed product embeddings
  -> product IDs
  -> CSV catalog metadata
  -> Streamlit results UI
```

## Tech Stack

- Python 3.11
- Streamlit
- Hugging Face Transformers
- PyTorch CPU inference
- FAISS CPU vector search
- Pandas / NumPy
- Git LFS for large vector artifacts

## Data And Artifacts

The deploy repo includes the runtime artifacts needed by the app:

- `Data/data/flipkart_com-ecommerce_sample.csv` - product catalog metadata
- `embeddings.npy` - precomputed MiniLM product embeddings
- `id_list.npy` - product IDs aligned to embedding rows
- `index` - generated FAISS index artifact

The app rebuilds the FAISS in-memory index from `embeddings.npy` on startup so deployment remains portable.

## Local Setup

```bash
git clone https://github.com/Vishnusige/Ecommerce-retrieval-Deploy.git
cd Ecommerce-retrieval-Deploy
git lfs pull
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local app at:

```text
http://localhost:8501
```

## Environment Variables

Defaults are production-safe for the included dataset. Override only when moving artifacts or changing model behavior.

```text
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
PRODUCT_DATA_PATH=Data/data/flipkart_com-ecommerce_sample.csv
EMBEDDINGS_PATH=embeddings.npy
ID_LIST_PATH=id_list.npy
TOP_K=6
REQUEST_TIMEOUT_SECONDS=2.5
```

## Deploy On Streamlit Community Cloud

1. Push this repository to GitHub with Git LFS artifacts available.
2. Open Streamlit Community Cloud and create a new app.
3. Select the deployment repository and `main` branch.
4. Set the app file to `app.py`.
5. Deploy.

## Deploy On Render

This repo includes `render.yaml`.

1. Create a new Render Blueprint from the GitHub repository.
2. Render runs `pip install -r requirements.txt`.
3. Render starts the app with:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## Regenerate Embeddings

Use this only when the catalog changes.

```bash
python rebuild_embeddings.py
```

Then validate that the files have matching row counts:

```bash
python -c "import numpy as np, pandas as pd; print(np.load('embeddings.npy').shape); print(len(np.load('id_list.npy', allow_pickle=True))); print(len(pd.read_csv('Data/data/flipkart_com-ecommerce_sample.csv')))"
```

## Interview Talking Points

- I precomputed embeddings for the product catalog so search requests do not recompute the whole dataset.
- The online path only embeds the user query, then FAISS performs nearest-neighbor lookup.
- The CSV acts as the product metadata store for this demo; MongoDB can replace it when write-heavy catalog management or API-backed product updates are required.
- Streamlit was chosen for a fast, inspectable full-stack demo UI with low deployment overhead.
- Large `.npy` and FAISS artifacts are managed through Git LFS to keep GitHub usable.

## License

This project is licensed under the [MIT License](LICENSE).
