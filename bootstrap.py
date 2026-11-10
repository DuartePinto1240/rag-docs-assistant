import subprocess
import sys

import chromadb

db = chromadb.PersistentClient(path="chroma_db")
try:
    n = db.get_collection("uv_docs_150").count()
except Exception:
    n = 0

if n == 0:
    print("Índice não encontrado, a construir...", flush=True)
    subprocess.run([sys.executable, "ingest.py"], check=True)
else:
    print(f"Índice pronto: {n} chunks", flush=True)