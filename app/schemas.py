from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


class FatwaHit(BaseModel):
    id: int
    question: str
    title: str
    link: str
    similarity: float
    categories: Optional[List[str]] = None


class ChatResponse(BaseModel):
    mode: str  # "exact" , "approx" , "none"
    exact_match: bool
    similarity: float
    answer: str
    matched_question: Optional[str] = None
    fatwa_link: Optional[str] = None
    related_fatwas: List[FatwaHit] = Field(default_factory=list)
