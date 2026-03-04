import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# --- Load environment variables ---
load_dotenv()

class Settings:
  """
  Configuration settings for MongoDB and embedding model
  This ensures all scripts use consistent connection and model settings
  """
  def __init__(self):

    # --- MongoDB setup ---
    self.mongodb_uri = os.getenv("MONGODB_URI")
    if not self.mongodb_uri:
      raise RuntimeError("MONGODB_URI is not set. Put it in your .env (or environment).")
    self.db_name = os.getenv("MONGODB_DB", "RAG")
    self.coll_name = os.getenv("MONGODB_COL", "chunks")

    #Create MongoDB client and collection handle
    self.client = MongoClient(self.mongodb_uri)
    self.db = self.client[self.db_name]
    self.col = self.db[self.coll_name]

    # --- Embedding model setup ---
    # all-MiniLM-L6-v2 outputs 384-dimensional vectors
    self.embedding_model_name = os.getenv(
        "EMB_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    self.embedder = SentenceTransformer(self.embedding_model_name)

  def summary(self):
    """Optional: quick info printout for debugging."""
    print(f"Connected to MongoDB → DB: {self.db_name}, Collection: {self.coll_name}")
    print(f"Using Embedding Model → {self.embedding_model_name}")

# Create a global instance for convenience
settings = Settings()