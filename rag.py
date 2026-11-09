import sys

from dotenv import load_dotenv
from openai import OpenAI

from search import retrieve

load_dotenv()
client = OpenAI()

CHAT_MODEL = "gpt-4o-mini"
MAX_DISTANCE = 0.6  # acima disto, consideramos que não há contexto relevante

SYSTEM_PROMPT = """You are an assistant that answers questions about uv, the Python package manager.
Answer ONLY using the context provided below. Do not use outside knowledge.
After each claim, cite the source file in square brackets, e.g. [concepts__cache.md].
If the context does not contain the answer, reply exactly: "I don't know based on the documentation."
Be concise."""


def build_context(chunks: list[dict]) -> str:
    parts = [f"[{c['source']}]\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(parts)


def answer(question: str, k: int = 4) -> dict:
    chunks = retrieve(question, k=k)
    relevant = [c for c in chunks if c["distance"] <= MAX_DISTANCE]

    if not relevant:
        return {"answer": "I don't know based on the documentation.", "sources": []}

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{build_context(relevant)}\n\nQuestion: {question}",
            },
        ],
        temperature=0,
        max_tokens=400,
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": sorted({c["source"] for c in relevant}),
    }


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How do I install a specific Python version with uv?"
    result = answer(question)
    print(f"Pergunta: {question}\n")
    print(result["answer"])
    print("\nFontes:", ", ".join(result["sources"]) or "nenhuma")