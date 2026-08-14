from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "qwen2.5-0.5b"


@dataclass
class DocumentChunk:
    source: str
    chunk_id: int
    text: str


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Run: pip install -r requirements.txt")
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return read_text_file(path)
    if suffix == ".pdf":
        return read_pdf_file(path)
    return ""


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)

    return chunks


def load_documents(folder: str = "knowledge_base") -> List[DocumentChunk]:
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    supported = {".txt", ".md", ".pdf"}
    chunks: List[DocumentChunk] = []

    for path in sorted(folder_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        text = read_document(path)
        for chunk_id, chunk in enumerate(chunk_text(text), start=1):
            chunks.append(DocumentChunk(source=path.name, chunk_id=chunk_id, text=chunk))

    if not chunks:
        raise RuntimeError(
            f"No supported documents were found in '{folder}'. Add .txt, .md, or .pdf files."
        )
    return chunks


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


class FoundryLocalRAG:
    """RAG pipeline using Microsoft Foundry Local models."""

    def __init__(self, knowledge_base_folder: str = "knowledge_base"):
        from foundry_local_sdk import Configuration, FoundryLocalManager

        self.documents = load_documents(knowledge_base_folder)

        print("Initializing Microsoft Foundry Local...")
        FoundryLocalManager.initialize(Configuration(app_name="local_rag_assistant"))
        self.manager = FoundryLocalManager.instance

        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = self.manager.catalog.get_model(EMBEDDING_MODEL)
        self.embedding_model.download(
            lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True)
        )
        print()
        self.embedding_model.load()
        self.embedding_client = self.embedding_model.get_embedding_client()

        print(f"Indexing {len(self.documents)} document chunks...")
        response = self.embedding_client.generate_embeddings([d.text for d in self.documents])
        self.document_embeddings = [item.embedding for item in response.data]

        print(f"Loading chat model: {CHAT_MODEL}")
        self.chat_model = self.manager.catalog.get_model(CHAT_MODEL)
        self.chat_model.download(
            lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True)
        )
        print()
        self.chat_model.load()
        self.chat_client = self.chat_model.get_chat_client()

    def retrieve(self, question: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        response = self.embedding_client.generate_embeddings([question])
        question_embedding = response.data[0].embedding

        scored = [
            (doc, cosine_similarity(question_embedding, embedding))
            for doc, embedding in zip(self.documents, self.document_embeddings)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def answer(self, question: str, top_k: int = 3) -> Tuple[str, List[Tuple[DocumentChunk, float]]]:
        retrieved = self.retrieve(question, top_k=top_k)
        context = "\n\n".join(
            f"[Source: {doc.source}, chunk {doc.chunk_id}]\n{doc.text}"
            for doc, _score in retrieved
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a local RAG assistant. Answer using ONLY the context below. "
                    "If the context does not contain the answer, say: "
                    "I do not have enough information in the local documents to answer that.\n\n"
                    f"CONTEXT:\n{context}"
                ),
            },
            {"role": "user", "content": question},
        ]

        pieces: List[str] = []
        for chunk in self.chat_client.complete_streaming_chat(messages):
            delta = chunk.choices[0].delta.content
            if delta:
                pieces.append(delta)
        return "".join(pieces).strip(), retrieved

    def close(self) -> None:
        try:
            self.embedding_model.unload()
        finally:
            self.chat_model.unload()


class DemoRAG:
    """Small offline demo mode for testing the app structure without downloading models."""

    def __init__(self, knowledge_base_folder: str = "knowledge_base"):
        self.documents = load_documents(knowledge_base_folder)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token.strip(".,:;!?()[]{}'\"-").lower() for token in text.split() if token.strip()}

    def retrieve(self, question: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        q_tokens = self._tokenize(question)
        scored: List[Tuple[DocumentChunk, float]] = []
        for doc in self.documents:
            d_tokens = self._tokenize(doc.text)
            score = len(q_tokens & d_tokens) / max(1, len(q_tokens))
            scored.append((doc, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def answer(self, question: str, top_k: int = 3) -> Tuple[str, List[Tuple[DocumentChunk, float]]]:
        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved or retrieved[0][1] == 0:
            return "I do not have enough information in the local documents to answer that.", retrieved
        best = retrieved[0][0]
        answer = (
            "Demo answer based on the most relevant local document chunk: "
            + best.text[:450]
            + ("..." if len(best.text) > 450 else "")
        )
        return answer, retrieved

    def close(self) -> None:
        pass
