from pathlib import Path
from typing import Iterable, List
from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag import get_vector_store, embed_docs
from config import get_settings
from utils import load_file_as_text


def _split_docs(texts: List[str]):
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    docs = []
    for idx, text in enumerate(texts):
        chunks = splitter.split_text(text)
        docs.extend(
            [
                {
                    "id": f"{idx}-{i}",
                    "text": chunk,
                    "metadata": {"chunk": i},
                }
                for i, chunk in enumerate(chunks)
            ]
        )
    return docs


def ingest_documents(project_id: str, file_paths: Iterable[Path]):
    store = get_vector_store(collection=f"project-{project_id}")
    texts = [load_file_as_text(p) for p in file_paths]
    docs = _split_docs(texts)
    logger.info(f"Ingesting {len(docs)} chunks for project {project_id}")
    embed_docs(store, docs)


def ingest_regulation_docs(file_paths: Iterable[Path]):
    store = get_vector_store(collection="regulations")
    texts = [load_file_as_text(p) for p in file_paths]
    docs = _split_docs(texts)
    logger.info(f"Ingesting {len(docs)} regulation chunks")
    embed_docs(store, docs)
