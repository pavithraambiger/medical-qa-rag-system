# 🏥 Medical Q&A System — RAG + LLM

A healthcare-focused Question Answering system built using **Retrieval Augmented Generation (RAG)** architecture with LLM integration. Users ask medical questions and the system retrieves relevant clinical context from a medical knowledge base to generate accurate, grounded answers.

---

## How It Works

1. Medical knowledge base is loaded and split into chunks
2. Chunks are converted to vector embeddings using HuggingFace sentence-transformers
3. Embeddings are stored in a FAISS vector store for fast similarity search
4. When a user asks a question, FAISS retrieves the top 3 most relevant chunks
5. Retrieved context + question is passed to an LLM (Llama3 via Groq) with a prompt template
6. LLM generates a grounded, context-aware answer

---

## Tech Stack

- **Language:** Python
- **RAG Framework:** LangChain
- **Vector Store:** FAISS
- **Embeddings:** HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **LLM:** Llama3-8b via Groq API (free)
- **Deployment:** Flask Web Application
- **Frontend:** HTML, CSS, JavaScript

---

## Diseases Covered

- Diabetes
- Hypertension
- Chronic Kidney Disease (CKD)
- Brain Stroke
- Liver Disease
- Lung Cancer
- Heart Disease
- Mental Health

---

## Project Structure

```
medical-qa-rag/
│
├── rag_pipeline.py       ← Core RAG logic (load, embed, retrieve, generate)
├── app.py                ← Flask web application
├── medical_data.txt      ← Medical knowledge base
├── requirements.txt      ← Dependencies
├── faiss_index/          ← Auto-created vector store (after first run)
└── templates/
    └── index.html        ← Web UI
```

---

## Setup & Run

### Step 1 — Get Free Groq API Key
Go to https://console.groq.com → Sign up → Create API Key (free)

### Step 2 — Set API Key
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the Flask app
```bash
python app.py
```

### Step 5 — Open browser
Go to: http://127.0.0.1:5000

---

## Sample Questions to Try

- What are the symptoms of diabetes?
- How is hypertension treated?
- What causes chronic kidney disease?
- What are the risk factors for brain stroke?
- How is liver disease diagnosed?
- What are lung cancer symptoms?

---

## Skills Demonstrated

- Generative AI (RAG architecture)
- LLM integration and prompt engineering
- Vector embeddings and semantic search (FAISS)
- NLP and text preprocessing
- LangChain framework
- Flask web deployment
- Healthcare domain application
