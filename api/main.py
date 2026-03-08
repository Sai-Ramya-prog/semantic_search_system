from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from utils.preprocessing import load_dataset
from embeddings.embedder import Embedder
from vector_db.faiss_store import VectorStore
from clustering.fuzzy_cluster import FuzzyCluster
from cache.semantic_cache import SemanticCache


from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


app = FastAPI()


class QueryRequest(BaseModel):
    query: str


print("Initializing system...")


documents, labels, label_names = load_dataset()

documents = documents[:5000]

print("Documents loaded:", len(documents))


print("Generating embeddings...")

embedder = Embedder()

embeddings = embedder.embed_documents(documents)

print("Embedding shape:", embeddings.shape)


print("Building FAISS index...")

dimension = embeddings.shape[1]

vector_db = VectorStore(dimension)

vector_db.add_documents(embeddings, documents)

print("FAISS ready")


print("Reducing embedding dimensions with PCA...")

pca = PCA(n_components=30)

reduced_embeddings = pca.fit_transform(embeddings)

reduced_embeddings = normalize(reduced_embeddings)

print("Reduced embedding shape:", reduced_embeddings.shape)


print("Training fuzzy clustering...")

cluster_model = FuzzyCluster(n_clusters=15)

cluster_model.fit(reduced_embeddings)

print("Clustering complete")




cache = SemanticCache(threshold=0.85)

print("System initialization complete")



@app.post("/query")
def query_api(req: QueryRequest):

    query = req.query

    query_vector = embedder.embed_query(query)

   
    entry, sim = cache.lookup(query_vector)

    if entry:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(sim),
            "result": entry["result"],
            "dominant_cluster": entry["cluster"]
        }

   
    results = vector_db.search(query_vector)

   
    query_reduced = pca.transform([query_vector])

    query_reduced = normalize(query_reduced)[0]

    cluster_probs = cluster_model.predict(query_reduced)

    dominant_cluster = int(np.argmax(cluster_probs))

   
    cache.add(
        query=query,
        vector=query_vector,
        result=results,
        cluster=dominant_cluster
    )

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": 0,
        "result": results,
        "dominant_cluster": dominant_cluster
    }



@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "Cache cleared"}