import hashlib
import re
from typing import Iterable, List, Protocol

import torch


class LanguageEncoder(Protocol):
    embedding_dim: int

    def encode(self, texts: Iterable[str]) -> torch.Tensor:
        ...


class HashingLanguageEncoder:
    """Deterministic frozen text encoder for mechanism-level supervision.

    This is intentionally dependency-free. It turns short language descriptions such as "red block pushes blue cube"
    into fixed-size embeddings that can be stored in replay/offline datasets as `language_embedding`.
    """

    def __init__(self, embedding_dim: int = 512, lowercase: bool = True) -> None:
        self.embedding_dim = embedding_dim
        self.lowercase = lowercase

    def _tokens(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return re.findall(r"[a-z0-9_]+", text)

    def encode_one(self, text: str) -> torch.Tensor:
        embedding = torch.zeros(self.embedding_dim, dtype=torch.float32)
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], byteorder="little") % self.embedding_dim
            sign = 1.0 if int.from_bytes(digest[4:], byteorder="little") % 2 == 0 else -1.0
            embedding[bucket] += sign
        norm = embedding.norm().clamp_min(1.0)
        return embedding / norm

    def encode(self, texts: Iterable[str]) -> torch.Tensor:
        return torch.stack([self.encode_one(text) for text in texts], dim=0)


class SentenceTransformerLanguageEncoder:
    """Frozen sentence-transformers backend for real semantic language embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install sentence-transformers to use this language encoder backend.") from exc
        self.model = SentenceTransformer(model_name)
        self.model.eval()
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    @torch.no_grad()
    def encode(self, texts: Iterable[str]) -> torch.Tensor:
        embeddings = self.model.encode(list(texts), convert_to_tensor=True, normalize_embeddings=True)
        return embeddings.float().cpu()


class HuggingFaceLanguageEncoder:
    """Frozen HuggingFace text encoder with mean pooling."""

    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers to use this language encoder backend.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: Iterable[str]) -> torch.Tensor:
        encoded = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return torch.nn.functional.normalize(pooled.float(), dim=-1).cpu()


def make_language_encoder(
    backend: str = "hashing",
    embedding_dim: int = 512,
    model_name: str | None = None,
) -> LanguageEncoder:
    if backend == "hashing":
        return HashingLanguageEncoder(embedding_dim=embedding_dim)
    if backend == "sentence_transformers":
        return SentenceTransformerLanguageEncoder(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    if backend == "huggingface":
        return HuggingFaceLanguageEncoder(model_name or "distilbert-base-uncased")
    raise ValueError(f"Unsupported language encoder backend: {backend}")
