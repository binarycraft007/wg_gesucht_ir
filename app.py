import os

from flask import Flask, render_template, request

from ir_engine import IREngine

app = Flask(__name__)

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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port)
