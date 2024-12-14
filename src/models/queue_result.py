from pydantic import BaseModel
from typing import Optional

class QueueResult(BaseModel):
    results: Optional[any]
    processed_frame: Optional[any]
    if_exists: bool