import sys

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"

db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_collection("uv_docs")


def retrieve(question: str, k: int = 4) -> list[dict]:
    """Devolve os k chunks mais parecidos com a pergunta."""
    query_embedding = client.embeddings.create(
        model=EMBEDDING_MODEL, input=question
    ).data[0].embedding

    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({"text": doc, "source": meta["source"], "distance": dist})
    return chunks


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How do I install a specific Python version with uv?"
    print(f"Pergunta: {question}\n")
    for i, c in enumerate(retrieve(question), start=1):
        print(f"{i}. [{c['source']}] distância={c['distance']:.3f}")
        print(f"   {c['text'][:200]}...\n")