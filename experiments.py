import json
import time

import chromadb
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from chunking import load_chunks
from eval_answers import JUDGE_PROMPT
from rag import SYSTEM_PROMPT, build_context

load_dotenv()
client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"
MAX_DISTANCE = 0.6
REFUSAL = "i don't know"

# Preços em USD por token. CONFIRMA na página de preços da OpenAI, porque mudam.
PRICE_IN = 0.15 / 1_000_000
PRICE_OUT = 0.60 / 1_000_000

# (palavras por chunk, overlap, k)
CONFIGS = [
    (150, 25, 3), (150, 25, 5),
    (300, 50, 3), (300, 50, 5),
    (500, 80, 3), (500, 80, 5),
]

db = chromadb.PersistentClient(path="chroma_db")


def embed(texts):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def get_collection(size, overlap):
    """Cria (ou reutiliza) uma coleção Chroma para este tamanho de chunk."""
    name = f"uv_docs_{size}"
    try:
        col = db.get_collection(name)
        if col.count() > 0:
            return col
        db.delete_collection(name)
    except Exception:
        pass
    col = db.create_collection(name, metadata={"hnsw:space": "cosine"})
    chunks = load_chunks(size, overlap)
    print(f"  A criar {name}: {len(chunks)} chunks")
    for i in range(0, len(chunks), 50):
        batch = chunks[i:i + 50]
        col.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"]} for c in batch],
            embeddings=embed([c["text"] for c in batch]),
        )
    return col


def run_question(col, question, k):
    """Pipeline completo, medindo só retrieval + geração (sem o juiz)."""
    start = time.perf_counter()
    query_emb = embed([question])[0]
    res = col.query(query_embeddings=[query_emb], n_results=k)
    retrieved = [
        {"text": d, "source": m["source"], "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    relevant = [c for c in retrieved if c["distance"] <= MAX_DISTANCE]

    tokens_in = tokens_out = 0
    context = ""
    if not relevant:
        text = "I don't know based on the documentation."
    else:
        context = build_context(relevant)
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0,
            max_tokens=400,
        )
        text = response.choices[0].message.content
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
    latency = time.perf_counter() - start

    return {
        "answer": text, "context": context, "retrieved": retrieved,
        "by_filter": not relevant, "latency": latency,
        "tokens_in": tokens_in, "tokens_out": tokens_out,
    }


def judge(q, result):
    prompt = JUDGE_PROMPT.format(
        question=q["question"], reference=q["reference_answer"],
        context=result["context"], answer=result["answer"],
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0, max_tokens=200,
    )
    return json.loads(response.choices[0].message.content)


def run_config(questions, size, overlap, k):
    col = get_collection(size, overlap)
    rows = []
    for q in questions:
        r = run_question(col, q["question"], k)
        refused = REFUSAL in r["answer"].lower()
        row = {
            "config": f"{size}w_k{k}", "id": q["id"], "answerable": q["answerable"],
            "refused": refused, "by_filter": refused and r["by_filter"],
            "hit": None, "correct": None, "faithful": None,
            "latency": r["latency"], "tokens_in": r["tokens_in"],
            "tokens_out": r["tokens_out"], "answer": r["answer"],
        }
        if q["answerable"]:
            correct_files = {q["expected_source"], *q.get("acceptable_sources", [])}
            row["hit"] = any(c["source"] in correct_files for c in r["retrieved"])
            if refused:
                row["correct"] = False
            else:
                v = judge(q, r)
                row["correct"] = bool(v.get("correct"))
                row["faithful"] = bool(v.get("faithful"))
        rows.append(row)
    return rows


def summarize(df):
    out = []
    for config, g in df.groupby("config", sort=False):
        ans = g[g["answerable"]]
        unans = g[~g["answerable"]]
        answered = ans[~ans["refused"]]
        cost = (g["tokens_in"] * PRICE_IN + g["tokens_out"] * PRICE_OUT).sum() / len(g)
        out.append({
            "config": config,
            "hit": f"{ans['hit'].mean():.1%}",
            "correct": f"{ans['correct'].mean():.1%}",
            "faithful": f"{answered['faithful'].mean():.1%}",
            "false_refusals": int(ans["refused"].sum()),
            "refusal_ok": f"{int(unans['refused'].sum())}/{len(unans)}",
            "latency_s": round(g["latency"].mean(), 2),
            "tokens_in": int(g["tokens_in"].mean()),
            "cost_per_1000_q_usd": round(cost * 1000, 3),
        })
    return pd.DataFrame(out)


if __name__ == "__main__":
    with open("eval/questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    all_rows = []
    for size, overlap, k in CONFIGS:
        print(f"\n>>> chunks={size} overlap={overlap} k={k}")
        all_rows += run_config(questions, size, overlap, k)
        df = pd.DataFrame(all_rows)  # guarda após cada configuração
        df.to_csv("eval/experiments_detail.csv", index=False, encoding="utf-8-sig")
        summarize(df).to_csv("eval/experiments_summary.csv", index=False)

    print("\n=== RESULTADOS ===")
    print(summarize(pd.DataFrame(all_rows)).to_string(index=False))