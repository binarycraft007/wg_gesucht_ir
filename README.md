# WG-Gesucht Information Retrieval System

A complete Information Retrieval (IR) system for searching apartment and room rental listings scraped from WG-Gesucht (translated to English). The project features a full NLP processing pipeline, dynamic phrase learning, two distinct retrieval models (TF-IDF and BM25), an evaluation framework, and a web-based user interface.

## 🌟 Features

*   **Robust NLP Pipeline**: Tokenization, lowercasing, stop-word removal, and lemmatization using `NLTK`.
*   **Dynamic Phrase Learning**: Automatically extracts and tokenizes highly associated multi-word phrases (e.g., `prenzlauer_berg`, `washing_machine`) using NLTK's `BigramCollocationFinder` and Likelihood Ratio metrics to improve semantic matching.
*   **Dual Retrieval Models**:
    *   **TF-IDF** (Vector Space Model) with Cosine Similarity.
    *   **BM25** (Probabilistic Model) via the `rank-bm25` library for improved length normalization and term frequency saturation.
*   **Contextual Search Snippets**: Dynamically generates result snippets centered around matched query terms with visual HTML `<mark>` highlighting.
*   **Web Interface**: A clean, responsive frontend built with HTML5 and Bootstrap 5, served via Flask.
*   **Evaluation Framework**: Included script to evaluate system performance against a curated ground truth dataset using metrics like Precision@K, Recall, MRR, and MAP.

## 🛠️ Architecture

1.  **Dataset**: ~2,100 listings in `dataset_en.json`. Documents are concatenated from title, property description, and area descriptions.
2.  **Indexing**: On startup, the engine reads the dataset, processes the text, learns dynamic multi-word phrases, and builds both in-memory TF-IDF and BM25 indices.
3.  **Retrieval**: Queries are processed through the exact same NLP and phrase-learning pipeline before being vectorized and scored against the document corpus.

## 🚀 Setup & Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for fast Python package management.

1.  **Clone the repository** (if applicable) and navigate to the project directory:
    ```bash
    cd /path/to/wg_gesucht_ir
    ```

2.  **Ensure you have `uv` installed**, then run the application directly (uv will handle dependencies automatically):
    ```bash
    uv run app.py
    ```
    *The first startup might take a few moments as it downloads necessary NLTK corpora (stopwords, wordnet) and builds the indices.*

3.  **Access the web app**: Open your browser and go to `http://127.0.0.1:5000`.

## 📊 Evaluation

To measure the effectiveness of the retrieval models, the project includes an evaluation script that tests the models against a pre-defined set of queries and known relevant documents (Ground Truth / QRELS).

Run the evaluation script:
```bash
uv run evaluate.py
```

**Metrics Calculated:**
*   **Precision@K**: The percentage of relevant documents in the top K results.
*   **Recall**: The fraction of total relevant documents successfully retrieved.
*   **Mean Reciprocal Rank (MRR)**: Evaluates how high the *first* relevant document is ranked.
*   **Mean Average Precision (MAP)**: A comprehensive measure of ranking quality across multiple queries.

*Note: With dynamic phrase learning, the BM25 model significantly outperforms the baseline TF-IDF model in this dataset.*

## 📁 Project Structure

*   `app.py`: Flask web server and routing.
*   `ir_engine.py`: The core search engine, containing the NLP pipeline, phrase learning, indexing, and retrieval logic.
*   `evaluate.py`: Script to benchmark retrieval performance.
*   `templates/index.html`: The frontend user interface.
*   `pyproject.toml` & `uv.lock`: Dependency configuration.
