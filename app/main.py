from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag import answer_question

app = FastAPI(title="Smart Meter FAQ Assistant (RAG)")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=3, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return answer_question(req.question, req.k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
