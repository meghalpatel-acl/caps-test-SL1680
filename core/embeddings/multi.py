import logging
import os
import time
from pathlib import Path

import numpy as np
from llama_cpp import Llama

from .base import BaseEmbeddingsModel

logger = logging.getLogger(__name__)

def download(repo_id: str, filename: str) -> Path:
    # ...implement or import your download logic here...
    # Should return a Path to the downloaded file
    raise NotImplementedError("Download function must be implemented.")

class MultiLMLlama(BaseEmbeddingsModel):
    def __init__(
        self,
        model_name: str = "granite",
        normalize: bool = False,
        n_threads: int | None = None
    ):
        # Model selection logic
        if model_name == "granite":
            self.model_path = download(
                repo_id="bartowski/granite-embedding-107m-multilingual-GGUF",
                filename="granite-embedding-107m-multilingual-Q8_0.gguf",
            )
        else:
            self.model_path = download(
                repo_id="mykor/paraphrase-multilingual-MiniLM-L12-v2.gguf",
                filename="paraphrase-multilingual-MiniLM-L12-118M-v2-Q8_0.gguf",
            )
        super().__init__(
            model_name,
            self.model_path,
            normalize
        )
        self.model = Llama(
            model_path=str(self.model_path),
            n_threads=n_threads,
            n_threads_batch=n_threads,
            embedding=True,
            verbose=False
        )
        logger.info(f"Loaded Llama.cpp multilingual embeddings model '{self.model_path}'")

    def generate(self, text: str) -> list[float]:
        st = time.time()
        embedding = self.model.embed(text, normalize=self.normalize)
        et = time.time()
        self._infer_times.append(et - st)
        if embedding is None:
            raise ValueError("No embedding returned")
        return embedding

if __name__ == "__main__":
    pass
