from chunking import DATA_DIR, load_chunks, split_into_chunks


def make_text(n_words: int) -> str:
    return " ".join(f"w{i}" for i in range(n_words))


def test_short_text_gives_single_chunk():
    assert len(split_into_chunks(make_text(50), max_words=150, overlap=25)) == 1


def test_empty_text_gives_no_chunks():
    assert split_into_chunks("") == []


def test_chunks_respect_max_words():
    for chunk in split_into_chunks(make_text(1000), max_words=150, overlap=25):
        assert len(chunk.split()) <= 150


def test_consecutive_chunks_overlap():
    chunks = split_into_chunks(make_text(300), max_words=150, overlap=25)
    assert chunks[0].split()[-25:] == chunks[1].split()[:25]


def test_no_words_are_lost():
    text = make_text(1000)
    chunks = split_into_chunks(text, max_words=150, overlap=25)
    covered = {w for c in chunks for w in c.split()}
    assert covered == set(text.split())


def test_load_chunks_from_real_docs():
    chunks = load_chunks()
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    assert all(c["text"].strip() for c in chunks)
    # todos os .md de data/ geram pelo menos um chunk
    assert {c["source"] for c in chunks} == {p.name for p in DATA_DIR.glob("*.md")}