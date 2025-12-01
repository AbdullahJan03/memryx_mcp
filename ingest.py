import os
import requests
from bs4 import BeautifulSoup
import lancedb
from sentence_transformers import SentenceTransformer
import git
import glob
import time
from urllib.parse import urljoin
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Configuration
DOCS_URL = "https://developer.memryx.com"
GITHUB_REPO_URL = "https://github.com/memryx/MemryX_eXamples.git"
LOCAL_REPO_PATH = "./MemryX_eXamples"
DB_PATH = "./memryx_knowledge_base"

# Initialize Embedding Model & DB
model = SentenceTransformer('BAAI/bge-m3')
db = lancedb.connect(DB_PATH)

def get_parent_child_chunks(text: str, source: str, doc_type: str):
    """
    Implements 'Parent Document Retrieval' (Small-to-Big).
    1. Parent Splitter: Creates large context blocks (e.g., full functions/classes).
    2. Child Splitter: Creates small, dense vectors from the parent.
    We search against 'child', but return 'parent'.
    """
    # Determine language based on doc_type
    language = Language.CPP if doc_type.lower() in ['cpp', 'c++', 'cc', 'h', 'hpp'] else Language.PYTHON
    
    # 1. Parent Splitting (Large Context)
    parent_splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=2000,
        chunk_overlap=200
    )
    
    # 2. Child Splitting (Search Index)
    # Using generic splitter is fine for small chunks
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    
    parents = parent_splitter.create_documents([text])
    processed_chunks = []
    
    for parent in parents:
        parent_content = parent.page_content
        children = child_splitter.create_documents([parent_content])
        
        for child in children:
            processed_chunks.append({
                "text": child.page_content,
                "parent_context": parent_content,
                "source": source,
                "type": doc_type
            })
    
    return processed_chunks

def scrape_web_docs():
    """Scrapes the main MemryX developer hub.

    This function now automatically discovers all tutorial pages listed from
    the tutorials index and scrapes them, so you don't need to add tutorial
    pages one-by-one.
    """

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
                
            return sorted(links)
        except Exception as e:
            print(f"Failed to fetch tutorial index {index_url}: {e}")
            return []

    docs = []
    pages.extend(get_tutorial_pages())
    for page in pages:
        url = urljoin(DOCS_URL, page)
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer"]):
                script.extract()
            
            # Clean text
            text = soup.get_text(separator=' ', strip=True)
            
            # Apply Parent-Child Splitting
            chunks = get_parent_child_chunks(text, url, "documentation")
            docs.extend(chunks)
            time.sleep(0.1)
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    return docs

def scrape_github_code():
    """
    Clones the repo if missing, then scrapes Python code 
    with Syntax-Aware Parent-Child chunking.
    """
    code_snippets = []
    
    # --- Check and Clone Repo ---
    if not os.path.exists(LOCAL_REPO_PATH):
        print(f"Repository not found at {LOCAL_REPO_PATH}.")
        print(f"Cloning from {GITHUB_REPO_URL}...")
        try:
            git.Repo.clone_from(GITHUB_REPO_URL, LOCAL_REPO_PATH)
            print("Cloning complete.")
        except Exception as e:
            print(f"Failed to clone repository: {e}")
            return []
    else:
        print(f"Repository found at {LOCAL_REPO_PATH}. Skipping clone.")

    # --- Process Files ---
    # Note: Using LOCAL_REPO_PATH for glob, not the URL
    print(f"Processing Python files in {LOCAL_REPO_PATH}...")
    for filepath in glob.glob(f"{LOCAL_REPO_PATH}/**/*.py", recursive=True):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            chunks = get_parent_child_chunks(content, filepath, "code")
            code_snippets.extend(chunks)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
    return code_snippets

def create_index():

    web_data = scrape_web_docs()
    code_data = scrape_github_code()
    all_data = web_data + code_data
    
    if not all_data:
        print("Error: No data to index. Exiting.")
        return

    print(f"Processing {len(all_data)} child chunks...")

    batch_size = 32
    vectors = []
    texts = [d['text'] for d in all_data]
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_vectors = model.encode(batch)
        vectors.extend(batch_vectors)
    
    data_with_vectors = []
    for i, item in enumerate(all_data):
        item['vector'] = vectors[i]
        data_with_vectors.append(item)
    
    try:
        tbl = db.create_table("memryx_docs", data=data_with_vectors, mode="overwrite")
        # Full Text Search on the CHILD text (for precision)
        tbl.create_fts_index("text", replace=True) 
    except Exception as e:
        print(f"Error creating table: {e}")

if __name__ == "__main__":
    create_index()