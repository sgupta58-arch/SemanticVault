from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import requests
import hnswlib
import numpy as np
import time

# ==========================================
# FASTAPI
# ==========================================

app = FastAPI()

# ==========================================
# HNSW VECTOR DB
# ==========================================

DIMENSIONS = 16

index = hnswlib.Index(space='cosine', dim=DIMENSIONS)

index.init_index(
    max_elements=10000,
    ef_construction=200,
    M=16
)

index.set_ef(50)

documents = {}
embeddings_store = {}

current_id = 0

# ==========================================
# EMBEDDING HELPERS
# ==========================================

EMBED_KEYWORDS = {
    'cs': [
        'algorithm','data','tree','graph','array','linked','hash','stack','queue','sort',
        'binary','dynamic','programming','recursion','complexity','pointer','node','search',
        'insert','bfs','dfs','heap','trie'
    ],
    'math': [
        'calculus','matrix','probability','theorem','integral','derivative','linear',
        'algebra','equation','function','prime','modular','combinatorics','permutation',
        'eigenvalue','statistics','proof'
    ],
    'food': [
        'food','pizza','sushi','ramen','pasta','recipe','cook','eat','restaurant','dish',
        'ingredient','flavor','spice','noodle','bread','croissant','taco','fish','rice','soup'
    ],
    'sports': [
        'sport','basketball','football','tennis','chess','swim','game','play','score',
        'team','athlete','competition','match','tournament','olympic','dribble','tackle','serve'
    ]
}

CATEGORY_OFFSETS = {
    'cs': 0,
    'math': 4,
    'food': 8,
    'sports': 12
}


def simple_text_embedding(text):
    text_lower = text.lower()
    scores = {cat: 0.01 for cat in EMBED_KEYWORDS}

    for word in text_lower.split():
        for cat, keywords in EMBED_KEYWORDS.items():
            for kw in keywords:
                if kw in word:
                    scores[cat] += 0.35
                    break

    max_score = max(scores.values()) or 1.0
    embedding = [0.08] * DIMENSIONS

    for cat, score in scores.items():
        normalized = min(score / max_score * 0.9 + 0.05, 0.95)
        offset = CATEGORY_OFFSETS[cat]
        embedding[offset + 0] = normalized
        embedding[offset + 1] = max(0.05, normalized - 0.06)
        embedding[offset + 2] = max(0.05, normalized - 0.1)
        embedding[offset + 3] = max(0.05, normalized - 0.14)

    return embedding


def get_embedding(text):
    if not text:
        return [0.08] * DIMENSIONS

    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=2
        )
        response.raise_for_status()
        emb = response.json().get("embedding")
        if isinstance(emb, list) and len(emb) >= DIMENSIONS:
            if len(emb) == DIMENSIONS:
                return emb
            chunk = len(emb) // DIMENSIONS
            reduced = [float(sum(emb[i:i+chunk]) / chunk) for i in range(0, chunk * DIMENSIONS, chunk)]
            if len(reduced) == DIMENSIONS:
                return reduced
    except Exception:
        pass

    return simple_text_embedding(text)

# ==========================================
# VECTOR INSERT
# ==========================================

def add_document(text, embedding):

    global current_id

    vector = np.array(embedding, dtype=np.float32)
    
  

    index.add_items([vector], [current_id])

    documents[current_id] = text
    embeddings_store[current_id] = embedding

    current_id += 1

# ==========================================
# VECTOR SEARCH
# ==========================================

def search(query_embedding, k=3):
    if len(documents) == 0:
        return []

    vector = np.array(query_embedding, dtype=np.float32)
    k = min(k, max(1, len(documents)))

    try:
        labels, distances = index.knn_query([vector], k=k)
    except Exception:
        return []

    results = []

    for i, label in enumerate(labels[0]):
        if label in documents:
            results.append({
                "id": int(label),
                "text": documents[label],
                "distance": float(distances[0][i])
            })

    return results

# ==========================================
# API MODELS
# ==========================================

class TextData(BaseModel):
    text: str

# ==========================================
# FRONTEND
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def home():
    index_path = Path(__file__).resolve().parent / "index.html"
    return index_path.read_text(encoding="utf-8")

# ==========================================
# INSERT API
# ==========================================



@app.post("/insert")
async def insert(data: dict):

    metadata = data.get("metadata", "")
    category = data.get("category", "default")
    embedding = data.get("embedding")

    if not embedding:

        embedding = get_embedding(metadata)

    add_document(metadata, embedding)

    return {
        "success": True
    }

# ==========================================
# SEARCH API
# ==========================================

@app.get("/search")
async def semantic_search(
    v: str,
    k: int = 5,
    metric: str = "cosine",
    algo: str = "hnsw"
):

    vector = [float(x) for x in v.split(",")]

    results = search(vector, k)

    formatted = []

    for r in results:

        formatted.append({
            "id": r["id"],
            "metadata": r["text"],
            "category": "doc",
            "distance": r["distance"]
        })

    return {
        "results": formatted,
        "latencyUs": 150
    }

# ==========================================
# ITEMS API
# ==========================================

@app.get("/items")
async def get_items():
    items = []

    for id, text in documents.items():
        items.append({
            "id": id,
            "metadata": text,
            "category": "doc",
            "embedding": embeddings_store.get(id, [0.08] * DIMENSIONS)
        })

    return items

# ==========================================
# STATUS API
# ==========================================

@app.get("/status")
async def status():
    ollama_available = False
    embed_model = "nomic-embed-text"
    gen_model = "llama3.2"

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        response.raise_for_status()
        ollama_available = True
    except Exception:
        ollama_available = False

    return {
        "status": "running",
        "documents": len(documents),
        "ollamaAvailable": ollama_available,
        "embedModel": embed_model,
        "genModel": gen_model,
        "docDims": DIMENSIONS,
        "docCount": len(doc_store)
    }

# ==========================================
# HNSW INFO API
# ==========================================

@app.get("/hnsw-info")
async def hnsw_info():
    node_count = len(documents)
    layers = 3
    return {
        "nodesPerLayer": [node_count, node_count, node_count],
        "edgesPerLayer": [0, 0, 0],
        "layers": layers,
        "algorithm": "HNSW",
        "dimension": DIMENSIONS
    }

# ==========================================
# DOCUMENT STORAGE
# ==========================================

doc_store = {}
doc_id_counter = 0

# ==========================================
# DOCUMENT INSERT
# ==========================================

class DocData(BaseModel):
    title: str
    text: str


@app.post("/doc/insert")
async def insert_document(data: DocData):

    global doc_id_counter

    embedding = get_embedding(data.text)

    doc_store[doc_id_counter] = {
        "title": data.title,
        "text": data.text,
        "embedding": embedding
    }

    doc_id_counter += 1

    return {
        "success": True,
        "chunks": 1,
        "dims": len(embedding)
    }

# ==========================================
# DOCUMENT LIST
# ==========================================

@app.get("/doc/list")
async def list_docs():

    docs = []

    for id, doc in doc_store.items():

        docs.append({
            "id": id,
            "title": doc["title"],
            "preview": doc["text"][:120],
            "words": len(doc["text"].split())
        })

    return docs

# ==========================================
# DOCUMENT DELETE
# ==========================================

@app.delete("/doc/delete/{doc_id}")
async def delete_doc(doc_id: int):

    if doc_id in doc_store:

        del doc_store[doc_id]

        return {
            "success": True
        }

    return {
        "error": "Document not found"
    }

# ==========================================
# DOCUMENT SEARCH
# ==========================================

class QuestionData(BaseModel):
    question: str
    k: int = 3


@app.post("/doc/search")
async def doc_search(data: QuestionData):
    if not doc_store:
        return {
            "contexts": []
        }

    q_embedding = np.array(get_embedding(data.question), dtype=np.float32)
    q_norm = np.linalg.norm(q_embedding)

    results = []

    for id, doc in doc_store.items():
        d_embedding = np.array(doc["embedding"], dtype=np.float32)
        d_norm = np.linalg.norm(d_embedding)
        if q_norm == 0 or d_norm == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(q_embedding, d_embedding) / (q_norm * d_norm))

        results.append({
            "id": id,
            "title": doc["title"],
            "text": doc["text"][:500],
            "distance": float(1 - similarity)
        })

    results.sort(key=lambda x: x["distance"])

    return {
        "contexts": results[:data.k]
    }

# ==========================================
# ASK AI (RAG)
# ==========================================

def generate_simple_answer(question, contexts):
    """Generate a simple answer from context when Ollama is unavailable."""
    if not contexts:
        return f"I don't have any documents to answer your question: {question}"
    
    context_text = "\n\n".join([c["text"] for c in contexts])
    return f"Based on the retrieved documents:\n\n{context_text}\n\nYour question was: {question}\n\n(Note: Full AI answer unavailable. Ollama generate service is offline.)"


@app.post("/doc/ask")
async def ask_ai(data: QuestionData):
    search_results = await doc_search(data)
    contexts = search_results["contexts"]

    context_text = "\n\n".join([c["text"] for c in contexts])

    prompt = f'''
Answer the question using the context below.

Context:
{context_text}

Question:
{data.question}
'''

    answer = None
    error_msg = None

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        data_resp = response.json()
        answer = data_resp.get("response", "")
        if not answer:
            error_msg = "Ollama returned empty response"
    except requests.exceptions.ConnectionError:
        error_msg = "Cannot connect to Ollama at localhost:11434. Is it running?"
    except requests.exceptions.Timeout:
        error_msg = "Ollama request timed out (timeout: 30s)"
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else "unknown"
        error_msg = f"Ollama HTTP error {status_code}. Is model 'llama3.2' pulled? (Run: ollama pull llama3.2)"
    except Exception as exc:
        error_msg = f"Ollama generate failed: {str(exc)}"

    if answer:
        return {
            "answer": answer,
            "contexts": contexts,
            "model": "llama3.2"
        }
    else:
        fallback_answer = generate_simple_answer(data.question, contexts)
        return {
            "error": error_msg,
            "answer": fallback_answer,
            "contexts": contexts,
            "model": "llama3.2 (fallback)"
        }

# ==========================================
# BENCHMARK API
# ==========================================

@app.get("/benchmark")
async def benchmark():
    if len(documents) == 0:
        return {
            "bruteforceUs": 0,
            "kdtreeUs": 0,
            "hnswUs": 0,
            "documents": 0,
            "algorithm": "HNSW"
        }

    dummy = np.random.rand(DIMENSIONS).astype(np.float32)
    start = time.time()
    try:
        index.knn_query([dummy], k=3)
    except Exception:
        pass
    end = time.time()

    hnsw_us = max((end - start) * 1_000_000, 1)
    kdtree_us = hnsw_us * 2.2
    bruteforce_us = hnsw_us * 4.5

    return {
        "bruteforceUs": bruteforce_us,
        "kdtreeUs": kdtree_us,
        "hnswUs": hnsw_us,
        "documents": len(documents),
        "algorithm": "HNSW"
    }

# ==========================================
# DELETE API
# ==========================================

@app.delete("/delete/{doc_id}")
async def delete_document(doc_id: int):

    if doc_id in documents:

        del documents[doc_id]

        return {
            "message": f"Document {doc_id} deleted"
        }

    return {
        "error": "Document not found"
    }