# SemanticVault

> Full-stack AI-powered semantic search and Retrieval-Augmented Generation (RAG) platform built with FastAPI, HNSW vector search, Ollama LLMs, and interactive vector visualization.



---

# 📌 Overview

SemanticVault is a local-first AI knowledge retrieval engine that combines:

* Vector databases
* Semantic similarity search
* HNSW indexing
* Retrieval-Augmented Generation (RAG)
* Interactive vector visualization
* Local LLM inference with Ollama

The project demonstrates how modern AI search systems like Pinecone, Weaviate, Chroma, and Perplexity-style retrieval systems work internally.

Built entirely using Python, FastAPI, HNSWLIB, JavaScript Canvas, and Ollama.

---

# ✨ Features

| Feature                | Description                                     |
| ---------------------- | ----------------------------------------------- |
| 🔍 Semantic Search     | Search by meaning instead of keywords           |
| 🧠 RAG Pipeline        | Ask AI questions grounded in your own documents |
| ⚡ HNSW Search          | Production-style ANN vector retrieval           |
| 📊 PCA Visualization   | Real-time 2D vector space projection            |
| 🤖 Ollama Integration  | Local embeddings + local LLM responses          |
| 📁 Document Management | Insert, list, and delete semantic documents     |
| 📈 Benchmarking        | Compare HNSW, KD-Tree, and Brute Force          |
| 🛡 Graceful Fallbacks  | Works even when Ollama is unavailable           |
| 🌐 REST API            | Full backend API for vectors and RAG            |
| 🎨 Interactive UI      | Modern AI dashboard with semantic visualization |

---

# 🧠 How It Works

```text
Your Query
    │
    ▼
Embedding Generation (nomic-embed-text)
    │
    ▼
HNSW Vector Search
    │
    ▼
Top-K Semantic Context Retrieval
    │
    ▼
Prompt Construction
    │
    ▼
LLM Generation (llama3.2)
    │
    ▼
AI Response
```

---

# 🏗️ Technical Architecture

## Backend (Python + FastAPI)

SemanticVault uses a REST-based backend built with FastAPI.

### Core Backend Features

* FastAPI server architecture
* RESTful API endpoints
* HNSW vector indexing using HNSWLIB
* Semantic similarity search
* Document chunk retrieval
* Local embedding generation
* AI response generation
* Robust error handling
* Benchmarking APIs
* Real-time vector insertion

---

## Vector Database & Search

### HNSW (Hierarchical Navigable Small World)

The project uses HNSWLIB for fast approximate nearest neighbor search.

Features:

* Cosine similarity search
* Efficient ANN retrieval
* Configurable top-k search
* Real-time insertion
* Logarithmic retrieval complexity

### Additional Algorithms

SemanticVault also includes:

* KD-Tree search
* Brute Force search

for benchmarking and algorithm comparison.

---

## RAG Pipeline

The Retrieval-Augmented Generation system works in 4 stages:

### 1. Document Embedding

Documents are embedded using:

```text
nomic-embed-text
```

via Ollama.

---

### 2. Context Retrieval

The query embedding is matched against stored vectors using HNSW search.

---

### 3. Prompt Construction

Relevant contexts are merged into a structured prompt.

---

### 4. AI Generation

The prompt is sent to:

```text
llama3.2
```

for grounded answer generation.

---

# 📊 Vector Visualization

The frontend includes a real-time semantic space visualizer.

Features:

* PCA dimensionality reduction
* 2D scatter plot
* Query highlighting
* Semantic clustering
* Distance mapping
* Interactive search exploration

This allows users to visually understand how embeddings cluster in semantic space.

---

# 🖥️ Frontend

Built using:

* HTML
* CSS
* JavaScript
* Canvas API

### Frontend Modules

| Module    | Purpose                         |
| --------- | ------------------------------- |
| Search    | Semantic vector querying        |
| Documents | Document insertion & management |
| Ask AI    | RAG-powered AI assistant        |
| Benchmark | Algorithm comparison            |

---

# ⚙️ Tech Stack

## Backend

* Python
* FastAPI
* hnswlib
* NumPy

## AI / ML

* Ollama
* `nomic-embed-text`
* `llama3.2`

## Frontend

* HTML5
* CSS3
* JavaScript
* Canvas API

---

# 📡 API Endpoints

## Vector APIs

| Method   | Endpoint       | Description              |
| -------- | -------------- | ------------------------ |
| `POST`   | `/insert`      | Insert semantic vectors  |
| `GET`    | `/search`      | Vector similarity search |
| `GET`    | `/items`       | Retrieve all vectors     |
| `DELETE` | `/delete/{id}` | Delete vector            |
| `GET`    | `/benchmark`   | Benchmark algorithms     |
| `GET`    | `/hnsw-info`   | HNSW graph stats         |
| `GET`    | `/status`      | Backend & Ollama status  |

---

## Document & RAG APIs

| Method   | Endpoint           | Description                |
| -------- | ------------------ | -------------------------- |
| `POST`   | `/doc/insert`      | Insert documents           |
| `GET`    | `/doc/list`        | List documents             |
| `DELETE` | `/doc/delete/{id}` | Delete document            |
| `POST`   | `/doc/search`      | Retrieve semantic contexts |
| `POST`   | `/doc/ask`         | AI-powered RAG querying    |

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/SemanticVault.git

cd SemanticVault
```

---

## 2. Create Virtual Environment

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🤖 Install Ollama

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

# ▶️ Run Project

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

---

# 📂 Project Structure

```text
SemanticVault/
│
├── app.py
├── index.html
├── requirements.txt
├── preview.png
└── README.md
```

---

# 🧪 Skills Demonstrated

* Full-stack development
* AI engineering
* Vector databases
* ANN search algorithms
* Retrieval-Augmented Generation
* LLM integration
* REST API design
* Semantic search systems
* PCA visualization
* Backend architecture
* Error handling & fallbacks

---

# 🔬 Algorithms Used

| Algorithm   | Purpose                             |
| ----------- | ----------------------------------- |
| HNSW        | Approximate nearest neighbor search |
| KD-Tree     | Exact low-dimensional retrieval     |
| Brute Force | Baseline vector comparison          |
| PCA         | Dimensionality reduction            |

---

# 🔮 Future Improvements

* PDF ingestion
* Persistent vector storage
* Authentication system
* Streaming AI responses
* GPU acceleration
* Multi-user workspaces
* Hybrid search (BM25 + vectors)
* Docker deployment
* Cloud deployment
* Chat memory

---

# 📈 Benchmarking

SemanticVault supports algorithm benchmarking between:

* HNSW
* KD-Tree
* Brute Force

allowing comparison of:

* latency
* retrieval efficiency
* scalability

---

# 🎯 Project Goals

This project was built to deeply understand:

* how vector databases work
* semantic search systems
* RAG pipelines
* local AI infrastructure
* ANN search algorithms
* production AI backend architecture

---

# 📄 License

MIT License

---

# 👨‍💻 Author

Swayam Gupta


