from pydantic import BaseModel
from typing import List

class QuestionRequest(BaseModel):
    question: str

class Source(BaseModel):
    id: int
    source: str
    text: str
    score: float

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Source]
