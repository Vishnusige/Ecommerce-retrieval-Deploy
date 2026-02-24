# Search Functionality for a Ecommerce site.
* Try it out here: https://ecommerce-retrieval-search.streamlit.app/
* View the doc and approach for this project here: https://1drv.ms/w/s!AnO5FdGErMSuh7VXK1wgpzwajPXOGw?e=fAh5Qh

## Overview
This project aims to enhance the search experience for an e-commerce platform, by implementing an efficient and accurate product retrieval system. The system uses natural language processing techniques and vector similarity search to match user queries with relevant products.

## Features
* Text-based product search
* Image retrieval based on text queries
* Utilizes product metadata for improved search accuracy
* Fast similarity search using FAISS (Facebook AI Similarity Search)
* Simple and intuitive web interface


## How it works
1. Data Preprocessing:
    * Cleans and combines relevant product information (name, description, category, specifications, brand)
    * Performs text preprocessing (removing HTML tags, lowercasing, removing punctuation and stopwords)

2. Embedding Generation:
    * Uses the 'gte-base-1.5' embedding model (768 dimensions)
    * Converts preprocessed text into dense vector representations

3. Similarity Search:
    * Utilizes FAISS for efficient storage and querying of embeddings
    * Implements L2 distance (Euclidean) for similarity measurement

4. Query Processing:
    * Applies the same preprocessing to user queries
    * Generates embeddings for the query
    * Retrieves top K most similar products using FAISS

5. Result Display:
    * Fetches and displays product images based on the retrieved results


## Tech Stack
* *python*: Primary programming language
* *pandas*: Data manipulation and CSV handling
* *matplotlib*: Data visualization
* *transformers*: Embedding model
* *pyTorch*: Tensor operations and GPU support
* *nltk*: Text preprocessing (stopwords, stemming)
* *faiss-cpu*: Vector store for similarity search
* *streamlit*: GUI and deployment


## Installation and Usage
* Clone the repository.
* Install all the dependencies from *requirements.txt* file. Run `!pip install -r requirements.txt` in the terminal.
* Run `streamlit run app.py` and the app will run on localhost.


## Future Improvements
* Implement multimodal search using vision-language models like CLIP
* Enable image-to-image and image-to-text queries
* Fine-tune image models (e.g., ResNet, ViT) on the product dataset
* Implement re-ranking for multimodal queries
* Create a manually curated test set for evaluation (using metrics like Recall@K)
* Integrate an LLM for handling malformed and multilingual queries


## Results
* Query: A Red skirt
![A red skirt](<Imgs/Screenshot 2024-08-09 081151.png>)
* Query: Football shoes
![Football shoes](<Imgs/Screenshot 2024-08-09 081638.png>)
* Query: Running shoes
![Running shoes](<Imgs/Screenshot 2024-08-09 081959.png>)
* Query: Superhero t-shirt
![Superhero t-shirt](<Imgs/Screenshot 2024-08-09 082126.png>)


## License
This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.

