import logging
from .hybrid_retriever import HybridRetriever
from .reranker import Reranker
from ..api.models import SearchResult, SearchResponse

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.reranker = Reranker()

    def search(
        self,
        query: str,
        filters: dict[str, str] | None = None,
        top_k: int = 5,
        prefetch_k: int = 20,
    ) -> SearchResponse:
        fallback = "none"
        active_filters = filters or {}

        candidates, retrieval_ms = self.retriever.retrieve(
            query=query,
            filters=active_filters,
            top_k=prefetch_k,
        )

        if not candidates:
            if active_filters:
                logger.warning(
                    "No candidates with filters=%s, retrying without filters",
                    active_filters,
                )
                fallback = "retried_without_filters"
                candidates, retry_ms = self.retriever.retrieve(
                    query=query,
                    filters={},
                    top_k=prefetch_k,
                )
                retrieval_ms += retry_ms

        if not candidates:
            logger.warning("No candidates after fallback — returning empty response")
            return SearchResponse(
                results=[],
                retrieval_ms=retrieval_ms,
                rerank_ms=0.0,
                total_ms=retrieval_ms,
                fallback="no_results",
            )

        try:
            reranked, rerank_ms = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
        except Exception as exc:
            logger.error("Reranker failed (%s) — falling back to retrieval order", exc)
            fallback = "reranker_error"
            reranked = [
                {"candidate": c, "rerank_score": 0.0}
                for c in candidates[:top_k]
            ]
            rerank_ms = 0.0

        # Log rank changes for monitoring
        original_ids = [c.id for c in candidates]
        for new_rank, item in enumerate(reranked):
            old_rank = original_ids.index(item["candidate"].id)
            if old_rank != new_rank:
                logger.info(
                    "rank_change id=%s old=%d new=%d",
                    item["candidate"].id,
                    old_rank,
                    new_rank,
                )

        results = [
            SearchResult(
                id=item["candidate"].id,
                text=(
                    item["candidate"].payload.get("name", "")
                    + " "
                    + item["candidate"].payload.get("description", "")
                ).strip(),
                score=item["candidate"].score,
                rerank_score=item["rerank_score"],
                payload=item["candidate"].payload,
            )
            for item in reranked
        ]

        total_ms = retrieval_ms + rerank_ms
        return SearchResponse(
            results=results,
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            total_ms=total_ms,
            fallback=fallback,
        )
