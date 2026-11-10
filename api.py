from fastapi import FastAPI, HTTPException
from openai import OpenAIError
from pydantic import BaseModel, Field

from rag import answer

app = FastAPI(
    title="uv docs assistant",
    description="Responde a perguntas sobre a documentação do uv, com fontes citadas.",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


# "def" normal (não async): o FastAPI corre-o numa thread,
# o que é o correto porque as chamadas à OpenAI são bloqueantes.
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = answer(request.question)
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Erro no serviço de LLM: {e}")
    return AskResponse(answer=result["answer"], sources=result["sources"])