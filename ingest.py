import os
import requests
from bs4 import BeautifulSoup
import lancedb
from sentence_transformers import SentenceTransformer
import git
import glob
import time
from urllib.parse import urljoin

# Configuration
DOCS_URL = "https://developer.memryx.com"
GITHUB_REPO = "https://github.com/memryx/MemryX_eXamples.git"
DB_PATH = "./memryx_knowledge_base"

# Initialize Embedding Model & DB
model = SentenceTransformer('all-MiniLM-L6-v2')
db = lancedb.connect(DB_PATH)

def scrape_web_docs():
    """Scrapes the main MemryX developer hub.

    This function now automatically discovers all tutorial pages listed from
    the tutorials index and scrapes them, so you don't need to add tutorial
    pages one-by-one.
    """

    # Keep a small list of non-tutorial pages we still want to index.
    pages = [
        "/api/accelerator/python.html",
        "/api/accelerator/cpp.html",
        "/get_started/overview.html",
        "/tools/neural_compiler.html",
        "/tools/simulator.html",
        "/runtime/usage/callbacks.html",
        "/runtime/usage/streams.html",
        "/runtime/usage/prepost.html",
        "/runtime/usage/multi_device.html",
        "/runtime/usage/multi_dfp.html",
    ]

    def get_tutorial_pages():
        """Fetch the tutorials index and return a deduplicated list of full tutorial URLs after #tutorials section."""
        index_url = urljoin(DOCS_URL, "/tutorials/tutorials.html")
        try:
            resp = requests.get(index_url, timeout=10)
            soup = BeautifulSoup(resp.content, 'html.parser')
            links = set()
            found_tutorials_section = False
            
            for a in soup.find_all('a', href=True):
                href = a['href']
                
                # Detect the #tutorials anchor
                if href == '#tutorials':
                    found_tutorials_section = True  
                    continue
                
                # Only collect links after #tutorials section
                if not found_tutorials_section:
                    continue
                
                # Skip anchors and external links
                if href.startswith('#') or href.startswith('http'):
                    continue
                
                # Clean fragment identifiers
                href_clean = href.split('#')[0]
                
                # Prepend tutorials/ if it's a relative path without leading slash
                if href_clean and not href_clean.startswith('/'):
                    href_clean = f"/tutorials/{href_clean}"
                
            print(links)
            print(f"Discovered {len(links)} tutorial pages.")
            return sorted(links)
        except Exception as e:
            print(f"Failed to fetch tutorial index {index_url}: {e}")
            return []

    docs = []

    # Add discovered tutorial pages (full URLs)
    tutorial_pages = get_tutorial_pages()
    for t in tutorial_pages:
        pages.append(t)

    for page in pages:

        url = page if page.startswith('http') else urljoin(DOCS_URL, page)
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.content, 'html.parser')
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator=' ', strip=True)
            docs.append({"text": text, "source": url, "type": "documentation"})
            # be polite to the server
            time.sleep(0.1)
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    return docs
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Split text into overlapping chunks to preserve context."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at paragraph/sentence boundaries
        if end < len(text):
            last_period = chunk.rfind('. ')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > chunk_size * 0.7:  
                end = start + break_point + 1
        
        chunks.append(text[start:end])
        start = end - overlap  # Overlap for context
    
    return chunks

def scrape_github_code():
    repo_path = "./memryx_examples_repo"
    code_snippets = []
    for filepath in glob.glob(f"{repo_path}/**/*.py", recursive=True):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Chunk with overlap
            chunks = chunk_text(content, chunk_size=1200, overlap=200)
            for i, chunk in enumerate(chunks):
                code_snippets.append({
                    "text": chunk,
                    "source": f"{filepath}#chunk{i}",
                    "type": "code",
                    "language": "python"
                })
    return code_snippets

def create_index():
    data = scrape_web_docs() + scrape_github_code()
    
    # Create vector embeddings
    print("Creating embeddings...")
    vectors = model.encode([d['text'] for d in data])
    
    data_with_vectors = []
    for i, item in enumerate(data):
        item['vector'] = vectors[i]
        data_with_vectors.append(item)
    
    # Create table
    try:
        tbl = db.create_table("memryx_docs", data=data_with_vectors, mode="overwrite")
        tbl.create_fts_index("text", replace=True)  # Enable keyword search
    except Exception as e:
        print(f"Error creating table: {e}")

if __name__ == "__main__":
    create_index()