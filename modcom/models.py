from pydantic import BaseModel
from typing import Optional, List


class ClassificationResponse(BaseModel):
    success: bool
    msg: str
    prediction: Optional[List[float]]
