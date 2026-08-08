"""Request/response models for the FastAPI backend."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # Bounded so a client cannot ask the server to serialize the whole corpus.
    top_k: int = Field(default=5, ge=1, le=20)


class FatwaHit(BaseModel):
    id: int
    question: str
    title: str
    link: str
    similarity: float
    categories: List[str] = Field(default_factory=list)


class Citation(BaseModel):
    """A `[n]` marker in the answer, resolved to the fatwa it points at."""

    marker: int
    id: int
    title: str
    link: str


class ChatResponse(BaseModel):
    # "direct" | "derived" | "abstain" — decided by the gate, not by a threshold.
    verdict: str
    answered: bool
    premise_sound: bool = True
    premise_issue: Optional[str] = None
    answer: str
    similarity: float = 0.0
    citations: List[Citation] = Field(default_factory=list)
    related_fatwas: List[FatwaHit] = Field(default_factory=list)
