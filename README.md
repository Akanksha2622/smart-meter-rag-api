\---

title: Smart Meter FAQ Assistant

sdk: docker

app\_port: 7860

\---



\# Smart Meter FAQ Assistant (RAG API)



Ask questions about electricity and smart-meter documents and get answers with cited sources.



> Work in progress: live deployment coming soon.



\*\*Stack:\*\* FastAPI, ChromaDB, Sentence-Transformers, Groq (gpt-oss-20b), pytest, GitHub Actions, Docker.



\## How it works



documents -> chunking -> embeddings (all-MiniLM-L6-v2) -> ChromaDB -> top-k retrieval -> LLM (answers only from the retrieved context) -> answer + sources



\## Run locally



Requires Python 3.11 or 3.12 and a free Groq API key from https://console.groq.com/keys



```bash

pip install -r requirements.txt

export GROQ\_API\_KEY=your\_key\_here    # Windows: set GROQ\_API\_KEY=your\_key\_here

python -m app.ingest

uvicorn app.main:app --reload

```



Open http://127.0.0.1:8000/docs and try `POST /ask`:



```json

{"question": "When is the evening peak demand period?", "k": 3}

```



\## Tests and evaluation



```bash

pip install pytest httpx

pytest -v

python -m eval.evaluate

```



\## Evaluation results



20 hand-labeled questions on 1 document. A hit means the answer keyword appears in the retrieved chunk (retrieval only, not answer quality).



| Chunk size (chars) | Chunks | hit@1 | hit@3 |

|---|---|---|---|

| 300 | 9 | 75% | 95% |

| 600 | 5 | 90% | 95% |

| 1000 | 3 | 85% | 100% |



Chunk size 600 gave the best hit@1 and is the default.



Limitations: evaluated on a single short document, so hit@3 at size 1000 is trivially high (only 3 chunks exist). Chunks are cut by character count and can split words; sentence-aware chunking is a possible improvement.



\## Project structure



\- `app/rag.py`: chunking, embedding, retrieval, LLM call

\- `app/main.py`: FastAPI endpoints (`GET /health`, `POST /ask`)

\- `eval/`: retrieval evaluation script and labeled questions

\- `tests/`: pytest tests

\- `Dockerfile`, `.github/workflows/ci.yml`: container and CI

