import argparse
from pathlib import Path

import numpy as np

from utils.language import make_language_encoder


def process_episode(path: Path, backend: str, embedding_dim: int, model_name: str | None, overwrite: bool) -> bool:
    data = np.load(path, allow_pickle=True)
    if "language_description" not in data:
        return False
    if "language_embedding" in data and not overwrite:
        return False

    encoder = make_language_encoder(backend=backend, embedding_dim=embedding_dim, model_name=model_name)
    descriptions = [str(item) for item in np.asarray(data["language_description"]).reshape(-1)]
    embeddings = encoder.encode(descriptions).numpy().reshape(*data["language_description"].shape, -1)

    updated = {key: data[key] for key in data.files}
    updated["language_embedding"] = embeddings.astype("float32")
    np.savez(path, **updated)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute language embeddings for Do-World NPZ episodes.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--backend", default="hashing", choices=["hashing", "sentence_transformers", "huggingface"])
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    count = 0
    for episode_path in Path(args.root).rglob("episode.npz"):
        if process_episode(episode_path, args.backend, args.embedding_dim, args.model_name, args.overwrite):
            count += 1
            print(f"wrote language_embedding: {episode_path}")
    print(f"processed {count} episodes")


if __name__ == "__main__":
    main()
