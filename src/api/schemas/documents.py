from pydantic import BaseModel
from typing import Optional

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None
