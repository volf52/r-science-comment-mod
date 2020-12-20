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
        # MLModel(name="spacy_ensemble", type="spacy", display_name="Spacy - BOW + CNN"),
        # MLModel(name="spacy_simple_cnn", type="spacy", display_name="Spacy - Simple CNN"),
        MLModel(name="spacy_bow", type="spacy", display_name="Spacy - BOW"),
        MLModel(
            name="simple_logistic",
            type="sklearn",
            display_name="Simple Logistic Classifier",
        ),
    ]


API_CONFIG = AppConfig()


@lru_cache
def get_app_settings() -> AppConfig:
    return API_CONFIG
