import sys
import numpy as np
from config import settings 
from typing import List, Dict 


# --- Search function ---
def search_similarity(query: str, top_k: int = 3, doc_id: str = None) -> List[Dict]:
    """
    1. Embeds the user query.
    2. Performs a vector similarity search in MongoDB.
    3. Returns the top_k most relevant chunks.
    """

    # --- Load the embedding model ---
    model = settings.embedder
    coll = settings.col

    # --- Embed the query ---
    # normalize_embeddings=True scales each vector so that its length (L2 Norm) is exactly 1
    query_embedding = model.encode(query, normalize_embeddings=True) 

    # Convert to Python list of float32 for MongoDB
    query_vector = np.asarray(query_embedding, dtype=np.float32).tolist()

    # Define the MongoDB $vectorSearch stage
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",  
                "path": "embedding",      # The field that contains the vectors
                "queryVector": query_vector,
                "numCandidates": 200,     # Number of candidates to consider
                "limit": top_k            # Number of results to return
            }
        },
        {
            # ADVANCED: If this is a Hierarchical "child" chunk, this stage automatically 
            # looks up the associated "parent" document in the same collection.
            "$lookup": {
                "from": coll.name, 
                "localField": "parent_id",
                "foreignField": "doc_id",
                "as": "parent_info"
            }
        },
        {
            # It tells MongoDB exactly which fields to return to Python (1 = include, 0 = exclude).
            # This prevents downloading the massive embedding arrays back to your app.
            "$project": {
                "_id": 0,
                "doc_id": 1,
                "strategy_used": 1,
                "metadata": 1, # Includes Markdown headers if structural chunking was used
                "score": { "$meta": "vectorSearchScore" }, # Include the similarity score
                
                # Logic: If 'parent_info' exists (meaning it's a child chunk), return the parent's 
                # large text block. Otherwise, just return the normal chunk's text.
                "text": {
                    "$cond": {
                        "if": { "$gt": [{ "$size": { "$ifNull": ["$parent_info", []] } }, 0] },
                        "then": { "$arrayElemAt": ["$parent_info.text", 0] },
                        "else": "$text"
                    }
                }
            }
        } 
    ]  
    
    # --- Execute and Return Results
    try:
        results = list(coll.aggregate(pipeline)) 
        print(f"Found {len(results)} relevant chunks")
        return results
    except Exception as e:
        print(f"Error performing vector search: {e}")
        print("Please ensure the index exists and is correctly configured.")
        return []


# --- Test the function directly ---
# if __name__ == "__main__":

#     if len(sys.argv) < 2:
#         print("Usage: python search.py \"<your_query_here>\"")
#         raise SystemExit(1)
    
#     query = sys.argv[1]
    
#     print(f"\nSearching for: '{query}'")
    
#     search_results = search_similarity(query, top_k=3)
    
#     if search_results:
#         print("\n--- Top Results ---")
#         for i, res in enumerate(search_results):
#             print(f"\nRank {i+1} (Score: {res['score']:.4f}):")
#             print(f"  Strategy: {res.get('strategy_used', 'N/A')}")
            
#             # Print Markdown metadata if it exists (for structural chunking)
#             if 'metadata' in res and res['metadata']:
#                 print(f"  Headers: {res['metadata']}")
                
#             print(f"  Text: {res.get('text', '')[:250]}...") # Print a snippet
#     else:
#         print("No results found")