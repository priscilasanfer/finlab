from fastapi import APIRouter

from api.models.rag import RAGRequest, RAGResponse
from api.routers.search import search_service
from api.services.rag import RAGService

router = APIRouter()

rag_service = RAGService(search_service=search_service)


@router.post("/rag", response_model=RAGResponse)
def rag(request: RAGRequest):
    return rag_service.generate_answers(request.query, request.limit)
