"""
Medical Q&A System — Flask Web App
"""

from flask import Flask, render_template, request, jsonify
from rag_pipeline import load_vectorstore, build_vectorstore, build_qa_chain, load_documents
import os

app = Flask(__name__)

print("Loading Medical Q&A System...")
if os.path.exists("faiss_index"):
    vectorstore = load_vectorstore()
else:
    chunks = load_documents("medical_data.txt")
    vectorstore = build_vectorstore(chunks)

qa_chain = build_qa_chain(vectorstore)
print("System ready!")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Please enter a question"}), 400
    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        sources = [doc.page_content[:200] for doc in result.get("source_documents", [])]
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
