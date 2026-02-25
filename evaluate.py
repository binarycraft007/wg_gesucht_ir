import json

from ir_engine import IREngine

# A small set of sample queries and their known relevant document IDs (Ground Truth)
QRELS = {
    "quiet room with balcony": [13104668, 12165369, 13046592],
    "central apartment charlottenburg": [3945147],
    "sublet in neukölln": [10777111, 13120841],
    "prenzlauer berg women only": [8644104],
    "maisonette neukölln": [13120841],
    "furnished short term mitte": [12474703],
}


def calculate_precision_at_k(retrieved_docs, relevant_docs, k):
    """Calculates Precision@K"""
    top_k_retrieved = [doc["offer_id"] for doc in retrieved_docs[:k]]
    relevant_retrieved = set(top_k_retrieved).intersection(set(relevant_docs))
    return len(relevant_retrieved) / k if k > 0 else 0.0


def calculate_recall(retrieved_docs, relevant_docs):
    """Calculates Total Recall"""
    if not relevant_docs:
        return 0.0
    retrieved_ids = [doc["offer_id"] for doc in retrieved_docs]
    relevant_retrieved = set(retrieved_ids).intersection(set(relevant_docs))
    return len(relevant_retrieved) / len(relevant_docs)


def evaluate_model(engine, model_name, queries, k=5):
    print(f"\n--- Evaluating Model: {model_name.upper()} ---")

    total_p_at_k = 0
    total_recall = 0

    for query, relevant_docs in queries.items():
        results = engine.search(query, model=model_name, top_k=20)

        p_at_k = calculate_precision_at_k(results, relevant_docs, k)
        recall = calculate_recall(results, relevant_docs)

        total_p_at_k += p_at_k
        total_recall += recall

    avg_p_at_k = total_p_at_k / len(queries)
    avg_recall = total_recall / len(queries)

    print("-" * 40)
    print(f"Average P@{k}: {avg_p_at_k:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    return avg_p_at_k, avg_recall


if __name__ == "__main__":
    print("Loading Data & Initializing Indices...")
    engine = IREngine("dataset_en.json")

    print("\nStarting Evaluation against mock Ground Truth (QRELS)...")

    # Evaluate TF-IDF
    evaluate_model(engine, model_name="tfidf", queries=QRELS, k=3)

    # Evaluate BM25
    evaluate_model(engine, model_name="bm25", queries=QRELS, k=3)
