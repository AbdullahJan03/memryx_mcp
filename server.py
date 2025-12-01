import os
import sys
from mcp.server.fastmcp import FastMCP
import lancedb
from sentence_transformers import SentenceTransformer,CrossEncoder

# --- CONFIGURATION with ABSOLUTE PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "memryx_knowledge_base")

# Check if DB exists before starting
if not os.path.exists(DB_PATH):
    sys.exit(1)

# Initialize Server
mcp = FastMCP("MemryX Helper")

try:
    db = lancedb.connect(DB_PATH)
    # We try to open the table immediately to fail fast if it's missing
    tbl = db.open_table("memryx_docs")
    embed_model = SentenceTransformer('BAAI/bge-m3')
    reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
except Exception as e:
    sys.exit(1)

@mcp.tool()
def search_memryx_docs(query: str) -> str:
    """
    Search MemryX documentation and code examples using SOTA Re-ranking.
    Use this to find SDK methods, compiler flags, or implementation details.
    
    Args:
        query: The specific programming question or concept
    """
    try:
        query_vec = embed_model.encode([query])[0]
        
        # Semantic search on Child chunks
        semantic_results = tbl.search(query_vec).limit(25).to_list()
        
        # Keyword search on Child chunks
        keyword_results = tbl.search(query, query_type="fts").limit(25).to_list()
        
        # We map by 'parent_context' to ensure we don't present the same parent block twice
        unique_candidates = {}
        
        def add_candidates(results):
            for res in results:
                key = res['parent_context'][:100]
                if key not in unique_candidates:
                    unique_candidates[key] = res

        add_candidates(semantic_results)
        add_candidates(keyword_results)
        
        candidates = list(unique_candidates.values())
        if not candidates:
            return "No matching documentation found."

        cross_inp = [[query, doc['parent_context']] for doc in candidates]
        scores = reranker.predict(cross_inp)
        
        scored_results = []
        for i, doc in enumerate(candidates):
            scored_results.append({
                "doc": doc,
                "score": scores[i]
            })
            
        # Sort by Cross-Encoder score (Descending)
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        top_results = scored_results[:5]
        
        response = f"Found {len(top_results)} highly relevant MemryX resources:\n\n"
        for item in top_results:
            res = item['doc']
            score = item['score']
            
            response += f"--- Source: {res['source']} (Relevance: {score:.4f}) ---\n"
            response += f"Type: {res['type']}\n"
            response += f"{res['parent_context']}\n\n" # Returning full Parent Context
            
        return response

    except Exception as e:
        return f"Error searching database: {str(e)}"

if __name__ == "__main__":
    mcp.run()