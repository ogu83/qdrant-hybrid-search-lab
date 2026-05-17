from fastapi import APIRouter, HTTPException
from .models import SearchRequest, SearchResponse
from ..retrieval.pipeline import RetrievalPipeline

router = APIRouter()
pipeline = RetrievalPipeline()


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    return pipeline.search(
        query=request.query,
        filters=request.filters,
        top_k=request.top_k,
    )
