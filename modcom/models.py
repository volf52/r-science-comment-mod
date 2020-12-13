from typing import List, Optional

from pydantic import BaseModel


class ClassificationResponse(BaseModel):
    success: bool
    msg: str
    prediction: Optional[List[float]]


class MLModel(BaseModel):
    name: str
    type: str
    display_name: str
