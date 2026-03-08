# Semantic Search System with Fuzzy Clustering and Semantic Cache

## Overview
This project implements a lightweight semantic search system built on the **20 Newsgroups dataset**.  
The system supports intelligent query understanding using **sentence embeddings**, **vector search**, **fuzzy clustering**, and a **semantic cache** to avoid redundant computation.

The goal of the system is to retrieve semantically relevant documents even when the user query is phrased differently.

Example:

Query A:space shuttle launch mission
Query B:The system recognizes that these queries are semantically similar and can reuse cached results.
The system recognizes that these queries are semantically similar and can reuse cached results.
# System Architecture
Dataset
↓
Text Preprocessing
↓
SentenceTransformer Embeddings
↓
FAISS Vector Database
↓
PCA Dimensionality Reduction
↓
Fuzzy Clustering (Gaussian Mixture Model)
↓
Semantic Cache
↓
FastAPI Service

---

# Dataset

The system uses the **20 Newsgroups dataset**:

https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

Dataset properties:

- ~20,000 documents
- 20 topic categories
- real-world noisy email discussions

Example categories:

- sci.space
- comp.graphics
- talk.politics.guns
- rec.sport.baseball
- soc.religion.christian

---

# Key Components

## 1. Text Preprocessing
Raw email documents contain headers and metadata.  
The preprocessing step:

- removes email headers
- removes special characters
- normalizes text

This ensures embeddings focus on meaningful semantic content.

---
## 2. Sentence Embeddings
Embeddings are generated using:
sentence-transformers
model: all-MiniLM-L6-v2

Why this model?

- lightweight
- strong semantic similarity performance
- suitable for real-time search systems

Each document becomes a **384-dimensional vector representation**.

---

## 3. Vector Database (FAISS)

FAISS is used to store document embeddings and perform fast similarity search.

Benefits:

- efficient nearest neighbor search
- scalable to large embedding collections
- optimized for vector retrieval
When a query arrives:
1. query is embedded
2. FAISS retrieves top similar documents.
---
## 4. Dimensionality Reduction (PCA)

Embedding dimension is reduced:

Why this model?

- lightweight
- strong semantic similarity performance
- suitable for real-time search systems

Each document becomes a **384-dimensional vector representation**.

---

## 5. Fuzzy Clustering

Documents are clustered using **Gaussian Mixture Models (GMM)**.

Unlike hard clustering (K-Means), GMM produces **probabilistic cluster membership**.

Example:
Cluster 3 → 0.72
Cluster 5 → 0.18
Cluster 9 → 0.10

This reflects real-world ambiguity where a document may belong to multiple topics.

The system returns the **dominant cluster** for each query.

---

## 6. Semantic Cache

Traditional caches only match exact queries.

Example:
Query A: space shuttle launch mission
Query B: nasa shuttle launch


A traditional cache treats these as different queries.

Our **semantic cache** instead:

1. embeds the query
2. computes cosine similarity with cached queries
3. if similarity > threshold (0.85)

then cached result is returned.

Benefits:

- avoids redundant computation
- improves response time
- adapts to natural language variations

---

## 7. FastAPI Service

The system is exposed as a REST API using **FastAPI**.

Endpoints:

### POST /query

Input:


{
"query": "space shuttle launch mission"
}


Output:


{
"query": "...",
"cache_hit": true,
"matched_query": "...",
"similarity_score": 0.91,
"result": [...],
"dominant_cluster": 3
}


---

### GET /cache/stats

Returns cache statistics:


{
"total_entries": 5,
"hit_count": 2,
"miss_count": 3,
"hit_rate": 0.40
}


---

### DELETE /cache

Clears the cache.

---

# Installation

Clone the repository:


git clone https://github.com/YOUR_USERNAME/semantic-search-system.git

cd semantic-search-system


Create virtual environment:


python -m venv venv


Activate environment:

Windows:


venv\Scripts\activate


Install dependencies:


pip install -r requirements.txt


---

# Run the API

Start the service:


uvicorn api.main:app --host 0.0.0.0 --port 8000


Open API documentation:


http://localhost:8000/docs


---

# Example Query


POST /query

{
"query": "space shuttle launch"
}


---

# Technologies Used

- Python
- FastAPI
- SentenceTransformers
- FAISS
- Scikit-learn
- NumPy

---

# Design Decisions

### Why FAISS?

Efficient large-scale vector similarity search.

### Why PCA?

Improves clustering stability and reduces dimensionality.

### Why Gaussian Mixture?

Supports **soft clustering**, which is required because documents may belong to multiple topics.

### Why Semantic Cache?

Avoids recomputing results for semantically similar queries.

---

# Future Improvements

- distributed vector database
- online clustering updates
- query embedding batching
- improved cache eviction strategies

---

# Author
Sai Ramya
After Adding README
Commit it:
git add README.md
git commit -m "Added project README"
git push
