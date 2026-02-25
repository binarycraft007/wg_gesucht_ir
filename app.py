from flask import Flask, jsonify, render_template, request

from ir_engine import IREngine

app = Flask(__name__)

# Initialize the IR Engine (loads dataset and builds index on startup)
print("Initializing IR Engine...")
engine = IREngine("dataset_en.json")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    model = request.args.get("model", "tfidf")

    if not query:
        return render_template("index.html", error="Please enter a search query.")

    try:
        # Perform the search
        results = engine.search(query, model=model, top_k=20)

        return render_template(
            "index.html",
            query=query,
            results=results,
            model=model,
            result_count=len(results),
        )

    except Exception as e:
        print(f"Search Error: {str(e)}")
        return render_template("index.html", error="An error occurred during search.")


@app.route("/api/search", methods=["GET"])
def api_search():
    """Endpoint for returning JSON results directly (for testing or frontend integration)"""
    query = request.args.get("q", "")
    model = request.args.get("model", "tfidf")

    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    results = engine.search(query, model=model, top_k=20)
    return jsonify(
        {
            "query": query,
            "model": model,
            "total_results": len(results),
            "results": results,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
