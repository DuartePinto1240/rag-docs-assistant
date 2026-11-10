import streamlit as st
import bootstrap  # constrói o índice na primeira execução, se não existir
from rag import answer

st.set_page_config(page_title="uv docs assistant", page_icon="📚")

MAX_QUESTIONS = 20  # limite por sessão, para proteger o crédito da OpenAI

EXAMPLES = [
    "How do I install a specific Python version with uv?",
    "my CI cache is getting huge, how do i shrink it",
    "How do I use uv in a Dockerfile?",
    "How do I use uv with Bazel?",
]

DOCS_BASE = "https://docs.astral.sh/uv/"


def source_link(filename: str) -> str:
    """concepts__projects__sync.md -> link para a página da documentação."""
    path = filename.removesuffix(".md").replace("__", "/")
    return f"[{filename}]({DOCS_BASE}{path}/)"


def use_example(text: str):
    st.session_state.question = text


if "count" not in st.session_state:
    st.session_state.count = 0

st.title("📚 uv docs assistant")
st.caption(
    "Perguntas sobre a documentação do uv, com respostas baseadas só nos "
    "documentos e com fontes citadas. Se a resposta não estiver lá, diz que não sabe."
)

st.write("**Exemplos:**")
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    col.button(example, on_click=use_example, args=(example,), use_container_width=True)

with st.form("ask_form"):
    question = st.text_input("A tua pergunta (em inglês)", key="question")
    submitted = st.form_submit_button("Perguntar", type="primary")

if submitted and question.strip():
    if st.session_state.count >= MAX_QUESTIONS:
        st.warning("Atingiste o limite de perguntas desta sessão.")
        st.stop()

    with st.spinner("A pesquisar na documentação..."):
        try:
            result = answer(question.strip())
        except Exception as e:
            st.error(f"Erro ao contactar o serviço: {e}")
            st.stop()

    st.session_state.count += 1

    st.subheader("Resposta")
    st.markdown(result["answer"])

    if result["sources"]:
        st.markdown("**Fontes:** " + " · ".join(source_link(s) for s in result["sources"]))

    if result["context"]:
        with st.expander("Ver os excertos enviados ao modelo"):
            st.text(result["context"])
    else:
        st.caption("Nenhum excerto suficientemente relevante foi encontrado, por isso o modelo nem foi chamado.")

st.sidebar.write(f"Perguntas nesta sessão: {st.session_state.count}/{MAX_QUESTIONS}")