# Traditional and Agentic RAG Pipeline

This repository contains two Jupyter notebooks that explore and implement **Retrieval-Augmented Generation (RAG)** from the ground up — starting with a traditional pipeline and evolving into an agentic workflow.

---

## 📂 Notebooks

### 1. `Rag_pipeline.ipynb` — Traditional RAG
A step-by-step implementation of a classic RAG pipeline built from scratch using LangChain and ChromaDB.

**What it covers:**
- Loading documents (`.txt` and `.pdf`) using `TextLoader`, `DirectoryLoader`, `PyMuPDFLoader`
- Splitting documents into chunks using `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 200)
- Generating embeddings using `SentenceTransformer` (`all-MiniLM-L6-v2`)
- Storing and retrieving embeddings using **ChromaDB** vector store
- Querying with a `RAGRetriever` class that uses cosine similarity
- Generating answers using **Groq LLM** (`llama-3.1-8b-instant`)
- Simple and advanced RAG functions with source tracking and confidence scores

---

### 2. `Rag_system.ipynb` — Agentic RAG with LangGraph + Typesense
Extends the traditional pipeline with an **agentic workflow** and integrates a **full-text search engine**.

**What it covers (in addition to the above):**
- Everything from `Rag_pipeline.ipynb` as the base
- **Typesense** integration for cloud-based full-text search over a books dataset (`.jsonl`)
  - Schema creation, document ingestion, and search with sorting/filtering
- **Agentic RAG with LangGraph**:
  - `AgentState` typed dictionary to manage workflow state
  - `decide_retrieval` node — decides if retrieval is needed based on the query
  - `retrieve_documents` node — fetches relevant docs from FAISS vector store
  - `generate_answer` node — generates response using retrieved context or directly
  - Conditional edges connecting nodes into a full stateful graph
- **FAISS** vector store with `HuggingFaceEmbeddings`
- **GPT-4.1** as the LLM for the agentic pipeline

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Framework | LangChain, LangGraph |
| LLMs | Groq (`llama-3.1-8b-instant`), OpenAI (`gpt-4.1`) |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`), HuggingFace |
| Vector Stores | ChromaDB, FAISS |
| Search Engine | Typesense |
| Document Loaders | PyPDF, PyMuPDF, LangChain Loaders |
| Platform | Google Colab |

---

## 🔐 API Keys Setup

This project uses **Google Colab Secrets** to manage API keys safely. No keys are hardcoded.

In your Colab notebook, click the 🔑 **key icon** in the left sidebar and add:

| Secret Name | Description |
|---|---|
| `groq_key` | Your Groq API key from [console.groq.com](https://console.groq.com) |
| `TYPESENSE_HOST` | Your Typesense cloud host URL |
| `TYPESENSE_API_KEY` | Your Typesense API key |

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Add your API keys to Colab Secrets (🔑 sidebar)
3. Run all cells from top to bottom
4. No local setup required

---

## 📌 Key Concepts Demonstrated

- **Traditional RAG**: Document loading → Chunking → Embedding → Vector search → LLM generation
- **Agentic RAG**: Stateful graph-based workflow that decides *when* to retrieve and *how* to respond
- **Full-text Search**: Typesense for keyword-based search as a complement to semantic search
