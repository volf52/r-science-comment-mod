from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import BaseConfig

from modcom.models import MLModel


class AppConfig(BaseConfig):
    model_dir: Path = Path(__file__).parent.joinpath("../models").resolve()
    model_ext: str = "ml"

    vectorizer: str = "vectorizer"

    available_models: List[MLModel] = [
        MLModel(
            name="simple_logistic",
            type="sklearn",
            display_name="Simple Logistic Classifier",
        ),
        MLModel(name="spacy_textcat", type="spacy", display_name="Spacy"),
    ]


API_CONFIG = AppConfig()


@lru_cache
def get_app_settings() -> AppConfig:
    return API_CONFIG
