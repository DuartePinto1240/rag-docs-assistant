import logging
import os


def setup_logging() -> None:
    """Configura o logging. O nível vem de LOG_LEVEL (por defeito INFO)."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Bibliotecas que falam demais
    for noisy in ("httpx", "httpx2", "httpcore", "chromadb", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)