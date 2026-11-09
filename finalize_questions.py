import json

with open("eval/candidates.json", encoding="utf-8") as f:
    data = json.load(f)

# 1. Remover a q17 (duplica a q22/q36 sobre dependency-metadata)
data = [q for q in data if q["id"] != "q17"]

# 2. Corrigir a q04 (o nome do pacote não é argumento do comando)
for q in data:
    if q["id"] == "q04":
        q["reference_answer"] = (
            "Use `uv version --bump minor`, which increments the minor component "
            "of the package version (e.g. 1.2.3 => 1.3.0)."
        )

# 3. Ficheiros alternativos que também respondem corretamente
ALTERNATIVES = {
    "q22": ["concepts__projects__config.md"],
    "q36": ["concepts__projects__config.md"],
    "q23": ["getting-started__features.md", "guides__install-python.md"],
    "q01": ["concepts__tools.md"],
    "q05": ["concepts__tools.md"],
}
for q in data:
    q["acceptable_sources"] = ALTERNATIVES.get(q["id"], [])

# 4. Perguntas escritas à mão (informais, como um utilizador real)
HAND_WRITTEN = [
    ("h01", "uvx vs uv tool install, what's the difference?",
     "uvx runs a tool in a temporary, isolated environment without installing it, while `uv tool install` installs it for repeated use.",
     "guides__tools.md"),
    ("h02", "in docker how do i install only the deps and skip my own project so layers cache better",
     "Use `uv sync --no-install-project`, which skips the current project but still installs all its dependencies.",
     "concepts__projects__sync.md"),
    ("h03", "my CI cache is getting huge, how do i shrink it",
     "Run `uv cache prune --ci` at the end of the job. It removes pre-built wheels and unzipped source distributions but keeps wheels built from source.",
     "concepts__cache.md"),
    ("h04", "can uv upgrade my python 3.12 to 3.13 on its own?",
     "No. uv only upgrades transparently to the latest patch release (e.g. 3.13.4 to 3.13.5), not across minor versions, because that can affect dependency resolution.",
     "concepts__python-versions.md"),
    ("h05", "where do i put a uv.toml to apply to all my projects on windows",
     "User-level configuration goes in `%APPDATA%\\uv\\uv.toml` on Windows (`~/.config/uv/uv.toml` on macOS and Linux).",
     "concepts__configuration-files.md"),
    ("h06", "install a dependency straight from a github tag with uv",
     "Use `uv add git+<repository-url> --tag <tag>`, which adds a `git` source with a `tag` under `tool.uv.sources`.",
     "concepts__projects__dependencies.md"),
    ("h07", "does uv support pip's --user flag?",
     "No. uv does not support the complete set of pip's options; `--user` is one of the options not supported.",
     "pip__compatibility.md"),
]

# inserir as perguntas à mão antes das não respondíveis
answerable = [q for q in data if q["answerable"]]
unanswerable = [q for q in data if not q["answerable"]]
for qid, question, answer, source in HAND_WRITTEN:
    answerable.append({
        "id": qid, "question": question, "reference_answer": answer,
        "expected_source": source, "acceptable_sources": [], "answerable": True,
    })
data = answerable + unanswerable

# 5. Remover o campo context e guardar
for q in data:
    q.pop("context", None)

with open("eval/questions.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"{len(data)} perguntas guardadas "
      f"({len(answerable)} respondíveis, {len(unanswerable)} sem resposta)")