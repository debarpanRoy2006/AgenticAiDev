# --- agent_app/views.py (Relevant section for ChromaDB initialization) ---

# ChromaDB and Sentence-Transformers imports
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client
# For a persistent local client (data is stored in a file):
# chroma_client = chromadb.PersistentClient(path="./chroma_db_data") 
# For an in-memory client (data is lost on restart - good for quick demos):
chroma_client = chromadb.Client() 

# Get or create a collection
collection_name = "agent_ai_interactions"
try:
    # CORRECTED LINE: Call get_or_create_collection on the client instance
    vector_db_collection = chroma_client.get_or_create_collection(name=collection_name)
except Exception as e:
    print(f"Error getting/creating ChromaDB collection: {e}")
    vector_db_collection = None # Set to None to prevent further errors

# Initialize embedding model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    embedding_model = None

# ... rest of your views.py code ...
