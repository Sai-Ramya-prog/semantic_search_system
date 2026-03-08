from sentence_transformers import SentenceTransformer


class Embedder:
    

    def __init__(self, model_name="all-MiniLM-L6-v2"):
     
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents):
       

        embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        return embeddings

    def embed_query(self, query):
       
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        return embedding[0]