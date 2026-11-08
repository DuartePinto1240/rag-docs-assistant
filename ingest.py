from dotenv import load_dotenv
from openai import OpenAI
import chromadb

from chunking import load_chunks

load_dotenv()
client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 50


def embed(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"A processar {len(chunks)} chunks...")

    db = chromadb.PersistentClient(path="chroma_db")
    # apaga a coleção antiga (se existir) para poder re-executar sem duplicados
    try:
        db.delete_collection("uv_docs")
    except Exception:
        pass
    collection = db.create_collection("uv_docs", metadata={"hnsw:space": "cosine"})

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"]} for c in batch],
            embeddings=embed([c["text"] for c in batch]),
        )
        print(f"  {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print("Concluído. Chunks guardados:", collection.count())