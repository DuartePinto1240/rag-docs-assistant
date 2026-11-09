import json

from search import retrieve

MAX_DISTANCE = 0.6
K_VALUES = [1, 3, 5]

with open("eval/questions.json", encoding="utf-8") as f:
    questions = json.load(f)

answerable = [q for q in questions if q["answerable"]]
unanswerable = [q for q in questions if not q["answerable"]]

hits = {k: 0 for k in K_VALUES}
reciprocal_ranks = []
misses = []

for q in answerable:
    results = retrieve(q["question"], k=max(K_VALUES))
    correct = {q["expected_source"], *q.get("acceptable_sources", [])}
    sources = [r["source"] for r in results]

    rank = next((i for i, s in enumerate(sources, start=1) if s in correct), None)
    reciprocal_ranks.append(1 / rank if rank else 0)
    for k in K_VALUES:
        if rank and rank <= k:
            hits[k] += 1
    if rank is None or rank > 3:
        misses.append((q["id"], q["question"], q["expected_source"], sources[:3]))

n = len(answerable)
print(f"Perguntas respondíveis: {n}")
for k in K_VALUES:
    print(f"Hit@{k}: {hits[k] / n:.1%}")
print(f"MRR: {sum(reciprocal_ranks) / n:.3f}")

# Recusa correta: o melhor chunk está acima do limite de distância
refused = 0
for q in unanswerable:
    best = retrieve(q["question"], k=1)[0]["distance"]
    ok = best > MAX_DISTANCE
    refused += ok
    print(f"  [{'OK ' if ok else 'FALHA'}] dist={best:.3f}  {q['question']}")
print(f"Recusa correta: {refused}/{len(unanswerable)}")

if misses:
    print("\nFalhas (fora do top 3):")
    for qid, question, expected, got in misses:
        print(f"  {qid}: {question}\n     esperado: {expected}\n     obtido:   {got}")