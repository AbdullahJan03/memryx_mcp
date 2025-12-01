# 🚀 MemryX MCP (Model Context Protocol) Server

This repository provides a custom **MemryX MCP server** that allows **GitHub Copilot** (or VS Code Copilot Agents) to query your locally running server through MCP.

---

## 📦 1. Create & Activate a Virtual Environment

```bash
python -m venv mxmcp
source mxmcp/bin/activate
```

---

## 📥 2. Install Required Dependencies

```bash
pip install mcp[cli] beautifulsoup4 requests lancedb sentence-transformers gitpython
```

These packages support:

- MCP server functionality  
- Web scraping  
- Embedding generation  
- Local database storage  
- Git operations  

---

## 📚 3. Ingest Documentation

This script crawls the MemryX Developer Hub and GitHub tutorials.

```bash
python ingest.py
```

---

## 🖥️ 4. Run & Test the MCP Server

```bash
python server.py
```

If successful, the server will start and wait for MCP connections.

---

## 🤖 5. Connect MCP to GitHub Copilot in VS Code

Create the VS Code configuration:

```bash
mkdir .vscode
touch .vscode/mcp.json
```

Add the MCP server config:

```json
{
    "servers": {
        "memryx": {
            "type": "stdio",
            "command": "/home/abdullah/memryx-mcp/.venv/bin/python",
            "args": [
                "/home/abdullah/memryx-mcp/server.py"
            ]
        }
    },
    "inputs": []
}
```

---

## ⚠️ Important: Update Paths

Make sure the following paths match your system:

- Path to the **Python inside your virtual environment**
- Path to **server.py**

You can verify using:

```bash
which python
realpath server.py
```

---

## ✅ Done!

After restarting VS Code, GitHub Copilot will automatically detect and connect to your custom MemryX MCP server.
