# E-commerce Product Retrieval System

## Overview
This project implements a semantic product retrieval system for e-commerce data.
The goal is to retrieve relevant products based on the meaning of user queries rather than exact keyword matches, enabling more flexible and context-aware search.

## Features
* Text-based semantic product search
* Retrieval of relevant products using vector similarity
* Uses product metadata for improved relevance
* Simple web interface for querying products


## Approach
1. Product text data is cleaned and preprocessed.
2. Text is converted into vector embeddings.
3. User queries are embedded into the same vector space.
4. Similarity between query and products is computed.
5. Top relevant products are returned based on similarity scores.


## Tech Stack
* Python
* Pandas
* NumPy
* NLP Embeddings
* Streamlit


## Installation and Usage
* Clone the repository.
* Install all the dependencies from *requirements.txt* file. Run `!pip install -r requirements.txt` in the terminal.
* Run `streamlit run app.py` and the app will run on localhost.


## License
This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.

