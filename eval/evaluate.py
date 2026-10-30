import json
from pathlib import Path

from app.rag import build_index, retrieve

questions = json.loads(Path("eval/questions.json").read_text())

print(f"{'chunk_size':>10} | {'chunks':>6} | {'hit@1':>5} | {'hit@3':>5}")
for size in (300, 600, 1000):
    n = build_index(chunk_size=size, overlap=size // 5)
    h1 = h3 = 0
    for q in questions:
        chunks = retrieve(q["question"], 3)
        kw = q["keyword"].lower()
        if kw in chunks[0]["text"].lower():
            h1 += 1
        if any(kw in c["text"].lower() for c in chunks):
            h3 += 1
    print(f"{size:>10} | {n:>6} | {h1/len(questions):>5.0%} | {h3/len(questions):>5.0%}")