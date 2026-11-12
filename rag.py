import logging
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

from logging_setup import setup_logging
from search import retrieve

load_dotenv()
client = OpenAI()
logger = logging.getLogger(__name__)

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


def answer(question: str, k: int = 5) -> dict:
    start = time.perf_counter()
    chunks = retrieve(question, k=k)
    relevant = [c for c in chunks if c["distance"] <= MAX_DISTANCE]
    best = min((c["distance"] for c in chunks), default=float("nan"))
    logger.info(
        "question=%r retrieved=%d relevant=%d best_distance=%.3f",
        question, len(chunks), len(relevant), best,
    )

    if not relevant:
        logger.info("refused_by_filter best_distance=%.3f threshold=%.2f", best, MAX_DISTANCE)
        return {"answer": "I don't know based on the documentation.", "sources": [], "context": ""}

    try:
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
    except Exception:
        logger.exception("llm_call_failed question=%r", question)
        raise

    text = response.choices[0].message.content
    refused = "i don't know" in text.lower()
    logger.info(
        "answered tokens_in=%d tokens_out=%d latency=%.2fs refused_by_model=%s",
        response.usage.prompt_tokens, response.usage.completion_tokens,
        time.perf_counter() - start, refused,
    )
    if refused:
        logger.warning("refused_by_model best_distance=%.3f question=%r", best, question)

    return {
        "answer": text,
        "sources": sorted({c["source"] for c in relevant}),
        "context": build_context(relevant),
    }


if __name__ == "__main__":
    setup_logging()
    question = " ".join(sys.argv[1:]) or "How do I install a specific Python version with uv?"
    result = answer(question)
    print(f"\nPergunta: {question}\n")
    print(result["answer"])
    print("\nFontes:", ", ".join(result["sources"]) or "nenhuma")