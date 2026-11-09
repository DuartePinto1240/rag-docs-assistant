import json
import random
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from chunking import load_chunks

load_dotenv()
client = OpenAI()

N_QUESTIONS = 40
MAX_PER_FILE = 2
MIN_WORDS = 120

PROMPT = """Below is an excerpt from the uv documentation (uv is a Python package manager).
Write ONE question a user might realistically ask, which is answered by this excerpt.
Rules:
- The question must be self-contained (never say "according to the text" or "in this excerpt").
- Do not copy the excerpt's phrasing too closely.
- Also write a short reference answer (1-2 sentences) based ONLY on the excerpt.
- If the excerpt is just links, headers or not useful, return {"skip": true}.
Return JSON: {"question": "...", "reference_answer": "..."}

Excerpt:
"""

UNANSWERABLE = [
    "What is the capital of France?",
    "How do I use uv with Bazel?",
    "How do I deploy a uv project to AWS Lambda?",
    "How do I configure uv with JFrog Artifactory?",
    "How do I set up Dependabot to update uv dependencies?",
    "How do I use uv with Coiled?",
    "How do I bake a chocolate cake?",
    "How do I write a for loop in Rust?",
]


def main():
    random.seed(42)
    chunks = [c for c in load_chunks() if len(c["text"].split()) >= MIN_WORDS]
    random.shuffle(chunks)

    per_file = Counter()
    items = []
    for chunk in chunks:
        if len(items) >= N_QUESTIONS:
            break
        if per_file[chunk["source"]] >= MAX_PER_FILE:
            continue

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PROMPT + chunk["text"]}],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=300,
        )
        data = json.loads(response.choices[0].message.content)
        if data.get("skip") or "question" not in data:
            continue

        per_file[chunk["source"]] += 1
        items.append({
            "id": f"q{len(items) + 1:02d}",
            "question": data["question"],
            "reference_answer": data["reference_answer"],
            "expected_source": chunk["source"],
            "answerable": True,
            "context": chunk["text"],  # só para te ajudar a rever; apaga depois
        })
        print(f"{len(items)}/{N_QUESTIONS} {chunk['source']}")

    for q in UNANSWERABLE:
        items.append({
            "id": f"q{len(items) + 1:02d}",
            "question": q,
            "reference_answer": "I don't know based on the documentation.",
            "expected_source": None,
            "answerable": False,
            "context": "",
        })

    Path("eval").mkdir(exist_ok=True)
    Path("eval/candidates.json").write_text(
        json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nGuardadas {len(items)} perguntas em eval/candidates.json")


if __name__ == "__main__":
    main()