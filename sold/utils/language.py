import hashlib
import re
from typing import Iterable, List

import torch


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
