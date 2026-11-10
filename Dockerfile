FROM python:3.14-slim

# Utilizador sem privilégios (o Hugging Face Spaces também exige isto)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user chunking.py ingest.py search.py rag.py bootstrap.py app.py ./
COPY --chown=user data/ data/

EXPOSE 7860

CMD ["sh", "-c", "python bootstrap.py && streamlit run app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true"]