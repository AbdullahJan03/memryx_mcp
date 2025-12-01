import os
import sys
from mcp.server.fastmcp import FastMCP
import lancedb
from sentence_transformers import SentenceTransformer

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
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    sys.exit(1)

def _merge_results(semantic_results, keyword_results, top_k=3, k=60):
    """
    Reciprocal Rank Fusion - combines rankings from different retrieval methods.
    
    Formula: score = sum(1 / (k + rank)) for each result list
    
    Args:
        k: constant (typically 60) to prevent division by zero and reduce impact of high ranks
    """
    scores = {}
    
    # Score semantic results
    for rank, result in enumerate(semantic_results, start=1):
        source = result['source']
        if source not in scores:
            scores[source] = {'result': result, 'score': 0}
        scores[source]['score'] += 1 / (k + rank)
    
    # Score keyword results
    for rank, result in enumerate(keyword_results, start=1):
        source = result['source']
        if source not in scores:
            scores[source] = {'result': result, 'score': 0}
        scores[source]['score'] += 1 / (k + rank)
    
    # Sort by combined score
    merged = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
    
    return [item['result'] for item in merged[:top_k]]

@mcp.tool()
def search_memryx_docs(query: str) -> str:
    """
    Search MemryX documentation and code examples.
    Use this to find SDK methods, compiler flags, or implementation details.
    
    Args:
        query: The specific programming question or concept to search for.
    """

    try:
        query_vec = model.encode([query])[0]
        semantic_results = tbl.search(query_vec).limit(5).to_list()
        keyword_results = tbl.search(query, query_type="fts").limit(5).to_list()
            
        results = _merge_results(semantic_results, keyword_results)

        response = "Found the following MemryX resources:\n\n"
        for res in results:
            response += f"--- Source: {res['source']} ---\n"
            response += f"{res['text'][:1500]}...\n\n"
        return response
    except Exception as e:
        return f"Error searching database: {str(e)}"

if __name__ == "__main__":
    mcp.run()