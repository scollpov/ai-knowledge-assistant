from pydantic import BaseModel
from typing import List
from typing import Optional, Dict

class QuestionRequest(BaseModel):
    question: str
    filter_metadata: Optional[Dict[str, str]] = None

class Source(BaseModel):
    id: int
    source: str
    text: str
    score: float

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Source]
