from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    filters: dict[str, str] = {}
    top_k: int = 5


class SearchResult(BaseModel):
    id: int
    text: str
    score: float
    rerank_score: float
    payload: dict


class SearchResponse(BaseModel):
    results: list[SearchResult]
    retrieval_ms: float
    rerank_ms: float
    total_ms: float
    fallback: str = "none"
