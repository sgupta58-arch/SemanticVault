# SemanticVault

> Full-stack AI-powered semantic search and Retrieval-Augmented Generation (RAG) platform built with FastAPI, HNSW vector search, and Ollama LLMs.

---

## 🚀 Project Overview

SemanticVault is a local-first AI knowledge retrieval system that enables semantic document search, vector similarity matching, and AI-powered question answering using Retrieval-Augmented Generation (RAG).

The system combines:

* Vector databases
* Approximate nearest neighbor search (HNSW)
* Embedding models
* Local LLM inference
* Interactive vector visualization

to create a fully functional semantic AI platform.

---

## ✨ Features

### 🔍 Semantic Search

* Search documents based on meaning instead of keywords
* Cosine similarity vector retrieval
* Configurable top-k nearest neighbor search

### 🧠 RAG (Retrieval-Augmented Generation)

* Ask questions grounded in uploaded documents
* Context-aware AI responses using local LLMs
* Automatic context retrieval and ranking

### ⚡ HNSW Vector Database

* High-performance ANN search using HNSWLIB
* Efficient vector indexing and retrieval
* Real-time similarity matching

### 📊 Vector Space Visualization

* Interactive 2D PCA projection of embeddings
* Real-time scatter plot visualization
* Query vector highlighting and distance mapping

### 🛠 Smart Fallback System

* Graceful degradation when Ollama is unavailable
* Local fallback embeddings generation
* Robust error handling and diagnostics

### 📁 Document Management

* Insert, delete, and list documents
* Metadata previews and word counts
* Semantic embedding storage

---

## 🏗️ Tech Stack

### Backend

* Python
* FastAPI
* hnswlib
* NumPy

### AI / ML

* Ollama
* `nomic-embed-text`
* `llama3.2`

### Frontend

* HTML
* CSS
* JavaScript
* Canvas API

---

## 🧠 System Architecture

```text
User Query
     ↓
Embedding Generation
     ↓
HNSW Vector Search
     ↓
Top-K Context Retrieval
     ↓
LLM Prompt Construction
     ↓
AI Response Generation
```

---

## 📡 API Endpoints

| Endpoint      | Description                 |
| ------------- | --------------------------- |
| `/insert`     | Insert semantic vectors     |
| `/search`     | Vector similarity search    |
| `/doc/insert` | Insert RAG documents        |
| `/doc/list`   | List stored documents       |
| `/doc/search` | Retrieve relevant contexts  |
| `/doc/ask`    | AI-powered RAG querying     |
| `/benchmark`  | Benchmark search algorithms |
| `/status`     | Backend + Ollama health     |
| `/hnsw-info`  | Vector DB statistics        |

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/semanticvault.git

cd semanticvault
```

---

### Create Virtual Environment

```bash
python3 -m venv venv

source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Install Ollama

Install [Ollama](https://ollama.com?utm_source=chatgpt.com)

Pull required models:

```bash
ollama pull nomic-embed-text

ollama pull llama3.2
```

Start Ollama:

```bash
ollama serve
```

---

## ▶️ Run Project

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

---

## 📊 Algorithms Implemented

* HNSW (Hierarchical Navigable Small World)
* Cosine Similarity Search
* KD-Tree Benchmarking
* Brute Force Similarity Search
* PCA Dimensionality Reduction

---

## 🧪 Skills Demonstrated

* Full-stack AI application development
* Vector databases & ANN search
* Retrieval-Augmented Generation (RAG)
* LLM integration
* Embedding systems
* REST API architecture
* Real-time visualization
* Error handling & fallback systems
* Semantic search engineering

---

## 🔮 Future Improvements

* PDF ingestion
* Multi-user authentication
* Persistent vector storage
* Streaming AI responses
* Hybrid search (BM25 + vectors)
* GPU acceleration
* Chat history memory
* Docker deployment

---

## 📸 Preview

```text
AI Semantic Search + Vector Visualization + Local RAG Assistant
```

---

## 📄 License

MIT License

---

## 👨‍💻 Author

Swayam Gupta


