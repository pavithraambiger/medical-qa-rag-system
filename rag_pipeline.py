"""
Medical Q&A System — RAG Pipeline
Uses HuggingFace embeddings + FAISS + Groq LLM (free API)
"""

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os

# ── Prompt Template ────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are a helpful medical assistant. Use the following medical context to answer 
the question clearly and accurately. If you don't know the answer from the context, 
say "I don't have enough information on this topic."

Context:
{context}

Question: {question}

Answer:
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ── Load & Chunk Documents ─────────────────────────────────────────────────────
def load_documents(file_path="medical_data.txt"):
    loader = TextLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"✅ Loaded {len(chunks)} chunks from medical knowledge base")
    return chunks

# ── Build FAISS Vector Store ───────────────────────────────────────────────────
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("✅ Vector store built and saved")
    return vectorstore

# ── Load Existing Vector Store ─────────────────────────────────────────────────
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("✅ Vector store loaded")
    return vectorstore

# ── Build QA Chain ─────────────────────────────────────────────────────────────
def build_qa_chain(vectorstore):
    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.2
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain

# ── Ask a Question ─────────────────────────────────────────────────────────────
def ask(qa_chain, question):
    result = qa_chain.invoke({"query": question})
    print(f"\n🔍 Question: {question}")
    print(f"💊 Answer: {result['result']}")
    return result['result']

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    # Build or load vector store
    if os.path.exists("faiss_index"):
        vectorstore = load_vectorstore()
    else:
        chunks = load_documents("medical_data.txt")
        vectorstore = build_vectorstore(chunks)

    qa_chain = build_qa_chain(vectorstore)

    # Test questions
    questions = [
        "What are the symptoms of diabetes?",
        "How is hypertension treated?",
        "What causes chronic kidney disease?",
        "What are the risk factors for brain stroke?",
        "How is liver disease diagnosed?"
    ]

    for q in questions:
        ask(qa_chain, q)
