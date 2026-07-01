from typing import List

from pydantic import BaseModel


class RAGRequest(BaseModel):
    query: str
    limit: int = 3


class RAGResponse(BaseModel):
    query: str
    answers: str
    metadata: List[dict]
