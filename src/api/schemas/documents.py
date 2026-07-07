from pydantic import BaseModel, Field
from typing import List, Optional


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None
    trace_id: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
