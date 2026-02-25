import json
import re

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords, wordnet
from nltk.metrics import BigramAssocMeasures
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def download_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
        nltk.download("punkt_tab")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")


class IREngine:
    def __init__(self, data_path="dataset_en.json"):
        download_nltk_resources()

        self.data_path = data_path
        self.documents = []
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        self.vectorizer = TfidfVectorizer(max_df=0.85, min_df=2)
        self.tfidf_matrix = None
        self.bm25_model = None

        self.load_and_index()

    def preprocess_text(self, text):
        if not text:
            return ""

        # Tokenization & Lowercasing
        tokens = nltk.word_tokenize(text.lower())

        # Stop-word Removal & Lemmatization
        # We filter out non-alphanumeric tokens as a basic cleaning step
        filtered_tokens = [
            self.lemmatizer.lemmatize(w)
            for w in tokens
            if w.isalnum() and w not in self.stop_words
        ]
        return " ".join(filtered_tokens)

    def load_and_index(self):
        print(f"Loading data from {self.data_path}...")
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.documents = raw_data

        corpus: list[str] = []

        print(f"Preprocessing {len(self.documents)} documents...")
        for doc in self.documents:
            title = doc.get("title", "")
            district = doc.get("district", "")

            doc["_raw_content"] = " ".join(
                filter(
                    None,
                    [
                        title,
                        doc.get("property_description", ""),
                        doc.get("area_description", ""),
                        doc.get("other_description", ""),
                    ],
                )
            )

            processed_text = self.preprocess_text(doc["_raw_content"])
            corpus.append(processed_text)

        print("Building TF-IDF Index...")
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        print("Building BM25 Index...")
        self.bm25_model = BM25Okapi([doc.split() for doc in corpus])
        print("Indexing Complete!")

    def generate_snippet(self, text, query_terms, snippet_length=200):
        """Generates a contextual snippet with highlighted query terms."""
        lower_text = text.lower()
        best_pos = 0

        # Find the first occurrence of any query term to center snippet
        for term in query_terms:
            if len(term) < 3:
                continue  # Skip very short words for finding center
            pos = lower_text.find(term.lower())
            if pos != -1:
                # Try to capture whole words by adjusting best_pos
                best_pos = max(0, pos - 50)
                break

        snippet = text[best_pos : best_pos + snippet_length]

        # Add ellipses to indicate truncated content
        if best_pos > 0:
            snippet = "..." + snippet
        if best_pos + snippet_length < len(text):
            snippet = snippet + "..."

        # Highlight terms (Case-insensitive)
        for term in query_terms:
            if len(term) > 2:  # Don't highlight very short words
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                snippet = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", snippet)

        return snippet

    def search(self, query, model="tfidf", top_k=10):
        # Preprocess the query to match indexed terms
        processed_query: list[str] = self.preprocess_text(query)
        processed_query_terms: list[list[str]] = processed_query.split()

        # Extract raw query terms for highlighting
        raw_query_terms = re.findall(r"\w+", query) + processed_query_terms

        if not processed_query_terms:
            return []

        if model == "tfidf":
            query_vec = self.vectorizer.transform([processed_query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

            # Get top indices and scores
            top_indices = similarities.argsort()[::-1][:top_k]
            scores = similarities[top_indices]

        elif model == "bm25":
            scores_raw = self.bm25_model.get_scores(processed_query_terms)
            top_indices = scores_raw.argsort()[::-1][:top_k]
            scores = scores_raw[top_indices]
        else:
            raise ValueError("Unknown model. Choose 'tfidf' or 'bm25'.")

        results = []
        for rank, (idx, score) in enumerate(zip(top_indices, scores)):
            if score <= 0.0:
                continue  # Skip completely irrelevant docs

            doc = self.documents[idx]
            snippet = self.generate_snippet(doc["_raw_content"], raw_query_terms)

            result = {
                "rank": rank + 1,
                "score": round(float(score), 4),
                "offer_id": doc.get("offer_id"),
                "title": doc.get("title"),
                "rent": doc.get("rent"),
                "size": doc.get("size"),
                "district": doc.get("district"),
                "available_from": doc.get("available_from"),
                "snippet": snippet,
            }
            results.append(result)

        return results


if __name__ == "__main__":
    # Quick test when running the engine directly
    engine = IREngine()
    print("Testing TF-IDF Search: 'quiet room in Mitte'")
    results = engine.search("quiet room in Mitte", model="tfidf", top_k=3)
    for r in results:
        print(f"Score: {r['score']} - {r['title']}")
