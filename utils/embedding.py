# embedding.py
from sentence_transformers import SentenceTransformer

# Load the embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    """Generate embeddings for a list of input strings."""
    return model.encode(texts).tolist()
