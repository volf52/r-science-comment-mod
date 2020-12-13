from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import BaseConfig, BaseModel


class ModelSettings(BaseModel):
    name: str
    type: str


class APIConfig(BaseConfig):
    model_dir: Path = Path(__file__).parent.joinpath("../models").resolve()
    model_ext: str = "ml"

    vectorizer: str = "vectorizer"

    available_models: List[ModelSettings] = [
        ModelSettings(name="simple_logistic", type="sklearn"),
        ModelSettings(name="spacy_textcat", type="spacy"),
    ]

API_CONFIG = APIConfig()

@lru_cache
def get_api_settings() -> APIConfig:
    return API_CONFIG
