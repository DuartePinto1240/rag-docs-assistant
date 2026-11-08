from pathlib import Path

DATA_DIR = Path("data")


def split_into_chunks(text: str, max_words: int = 300, overlap: int = 50) -> list[str]:
    """Divide o texto em chunks de ~max_words palavras, com sobreposição."""
    words = text.split()
    chunks = []
    step = max_words - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start:start + max_words]
        chunks.append(" ".join(chunk_words))
        if start + max_words >= len(words):
            break
    return chunks


def load_chunks(max_words: int = 300, overlap: int = 50) -> list[dict]:
    """Lê todos os .md de data/ e devolve uma lista de chunks com metadados."""
    all_chunks = []
    for path in sorted(DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(split_into_chunks(text, max_words, overlap)):
            all_chunks.append({
                "id": f"{path.stem}__{i}",
                "source": path.name,
                "text": chunk,
            })
    return all_chunks


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Total de chunks: {len(chunks)}")
    print(f"Ficheiros: {len({c['source'] for c in chunks})}")
    print("\n--- Exemplo (chunk 10) ---")
    print("Fonte:", chunks[10]["source"])
    print(chunks[10]["text"][:500])