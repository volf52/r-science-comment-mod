from typing import List, Optional

from pydantic import BaseModel


class ClassificationResponse(BaseModel):
    success: bool
    msg: str
    prediction: Optional[List[float]]
