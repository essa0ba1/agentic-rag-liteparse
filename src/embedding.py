import onnxruntime as ort
import numpy as np
from chonkie.embeddings import BaseEmbeddings

class ONNXEmbeddings(BaseEmbeddings):
    def __init__(self, model_path, tokenizer, providers=None):
        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = 4

        self.session = ort.InferenceSession(
            model_path,
            providers=providers or ["CPUExecutionProvider"],
            session_options=session_options
        )
        self.tokenizer = tokenizer

    @property
    def dimension(self) -> int:
        return self.session.get_outputs()[0].shape[-1]

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        outputs = self.session.run(None, dict(enc))
        # mean-pool token embeddings -> sentence vector, then normalize
        embeddings = outputs[0].mean(axis=1)
        return list(embeddings)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        return [self.count_tokens(t) for t in texts]

    def get_tokenizer(self):
        return self.tokenizer

    @classmethod
    def is_available(cls) -> bool:
        return True

    def __repr__(self) -> str:
        return "ONNXEmbeddings()"