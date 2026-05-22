import time
from sentence_transformers import CrossEncoder
from ..config import RERANK_MODEL


class Reranker:
    def __init__(self) -> None:
        self.model = CrossEncoder(RERANK_MODEL)

    @staticmethod
    def _candidate_text(payload: dict) -> str:
        parts = [
            payload.get("name", ""),
            payload.get("sku", ""),
            payload.get("category", ""),
            payload.get("storage", ""),
            payload.get("color", ""),
            payload.get("description", ""),
        ]
        return " ".join(str(p).strip() for p in parts if str(p).strip())

    def rerank(
        self,
        query: str,
        candidates: list,
        top_k: int = 5,
    ) -> tuple[list[dict], float]:
        t0 = time.perf_counter()

        texts = [self._candidate_text(c.payload) for c in candidates]
        pairs = [(query, text) for text in texts]
        scores = self.model.predict(pairs).tolist()

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for candidate, score in ranked[:top_k]:
            results.append(
                {
                    "candidate": candidate,
                    "rerank_score": score,
                }
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return results, elapsed_ms
