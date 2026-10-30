from app.rag import build_index

if __name__ == "__main__":
    n = build_index(chunk_size=600, overlap=120)
    print(f"Indexed {n} chunks.")