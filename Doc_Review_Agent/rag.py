from pathlib import Path
from typing import List, Dict
import hashlib
from loguru import logger
import chromadb
from chromadb.utils import embedding_functions
from config import get_settings


def _default_embedder():
    settings = get_settings()
    if settings.openai_api_key:
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key, model_name=settings.embedding_model
        )
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )


def get_vector_store(collection: str):
    settings = get_settings()
    Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=settings.chroma_path)
    embedder = _default_embedder()
    return client.get_or_create_collection(collection_name=collection, embedding_function=embedder)


def embed_docs(store, docs: List[Dict]):
    ids = []
    texts = []
    metadatas = []
    for doc in docs:
        text = doc["text"]
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        ids.append(doc.get("id") or digest)
        texts.append(text)
        metadatas.append(doc.get("metadata", {}))
    store.add(ids=ids, documents=texts, metadatas=metadatas)
    logger.info(f"Inserted {len(ids)} docs into collection {store.name}")


def similarity_search(store, query: str, top_k: int = 8):
    result = store.query(query_texts=[query], n_results=top_k)
    items = []
    for doc, meta, score in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        items.append({"text": doc, "metadata": meta, "score": score})
    return items
