import json

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from rag import answer

load_dotenv()
client = OpenAI()

JUDGE_MODEL = "gpt-4o-mini"
REFUSAL = "i don't know"

JUDGE_PROMPT = """You are grading an answer produced by a documentation assistant.

Question: {question}

Reference answer (ground truth): {reference}

Retrieved context given to the assistant:
{context}

Assistant's answer: {answer}

Grade two things:
1. "correct": true if the assistant's answer conveys the key information of the reference answer and does not contradict it. Different wording is fine.
2. "faithful": true if every factual claim in the assistant's answer is supported by the retrieved context. false if it adds details that are not in the context.

Return JSON only: {{"correct": true/false, "faithful": true/false, "reason": "one short sentence"}}"""


def judge(q: dict, result: dict) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=q["question"],
        reference=q["reference_answer"],
        context=result["context"],
        answer=result["answer"],
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=200,
    )
    return json.loads(response.choices[0].message.content)


def main():
    with open("eval/questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    rows = []
    for i, q in enumerate(questions, start=1):
        result = answer(q["question"])
        refused = REFUSAL in result["answer"].lower()
        row = {
            "id": q["id"],
            "question": q["question"],
            "answerable": q["answerable"],
            "answer": result["answer"],
            "refused": refused,
            "refused_by_filter": refused and not result["sources"],
            "cites_source": ".md]" in result["answer"],
            "correct": None,
            "faithful": None,
            "reason": "",
        }
        if q["answerable"] and not refused:
            verdict = judge(q, result)
            row["correct"] = bool(verdict.get("correct"))
            row["faithful"] = bool(verdict.get("faithful"))
            row["reason"] = verdict.get("reason", "")
        elif q["answerable"] and refused:
            row["correct"] = False
            row["reason"] = "Refused an answerable question"
        rows.append(row)
        print(f"{i}/{len(questions)} {q['id']}")

    df = pd.DataFrame(rows)
    df.to_csv("eval/results_answers.csv", index=False, encoding="utf-8-sig")

    ans = df[df["answerable"]]
    unans = df[~df["answerable"]]
    answered = ans[~ans["refused"]]

    print("\n=== PERGUNTAS RESPONDÍVEIS ===")
    print(f"Total: {len(ans)}")
    print(f"Correção:   {ans['correct'].mean():.1%}  (recusas indevidas contam como erro)")
    print(f"Fidelidade: {answered['faithful'].mean():.1%}  (só nas que foram respondidas)")
    print(f"Recusas indevidas: {int(ans['refused'].sum())}")
    print(f"Com citação de fonte: {answered['cites_source'].mean():.1%}")

    print("\n=== PERGUNTAS SEM RESPOSTA ===")
    print(f"Recusa correta: {int(unans['refused'].sum())}/{len(unans)}")
    print(f"  pelo filtro de distância: {int(unans['refused_by_filter'].sum())}")
    print(f"  pelo modelo (prompt):     {int((unans['refused'] & ~unans['refused_by_filter']).sum())}")
    for _, r in unans[~unans["refused"]].iterrows():
        print(f"  [INVENTOU] {r['question']}\n      -> {r['answer'][:150]}")

    bad = ans[(ans["correct"] == False) | (ans["faithful"] == False)]
    if len(bad):
        print("\n=== RESPONDÍVEIS COM PROBLEMAS ===")
        for _, r in bad.iterrows():
            print(f"  {r['id']}: {r['question']}\n      correct={r['correct']} faithful={r['faithful']} | {r['reason']}")

    print("\nResultados completos em eval/results_answers.csv")


if __name__ == "__main__":
    main()