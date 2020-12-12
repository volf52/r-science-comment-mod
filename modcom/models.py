from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union, Optional

import joblib
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .config import ModelSettings, APIConfig
from .ml import TextPreprocessor

CLASSIFIER = Union[LogisticRegression, SVC]
VECTORIZER = Union[TfidfVectorizer, CountVectorizer]

text_preproc = TextPreprocessor()


class Model(ABC):
    @abstractmethod
    def predict(self, clean_comment: str) -> List[float]:
        pass


class ScikitModel(Model):
    __slots__ = "_clf", "_vectorizer"

    def __init__(self, clf_path: Union[str, Path], *, vectorizer: VECTORIZER):
        self._clf: CLASSIFIER = joblib.load(clf_path)
        self._vectorizer = vectorizer

    def predict(self, clean_comment: str) -> List[float]:
        vec = text_preproc.clean_and_tokenize_single(clean_comment)
        vec = self._vectorizer.transform([vec])
        prob = self._clf.predict_proba(vec)

        return list(prob[0])


class SpacyModel(Model):
    POSITVE = "REMOVED"
    NEGATIVE = "NOTREMOVED"

    __slots__ = ("_nlp",)

    def __init__(self, path: Union[str, Path]):
        super().__init__()
        self._nlp = spacy.load(path)

    def predict(self, clean_comment: str) -> List[float]:
        doc: spacy.language.Doc = self._nlp(clean_comment)

        categories = doc.cats
        probs = [categories[self.NEGATIVE], categories[self.POSITVE]]

        return probs


class ModelLoader:
    __slots__ = "_settings", "_models", "_vec"

    def __init__(self, settings: APIConfig, vectorizer: VECTORIZER = None):
        self._settings = settings
        self._models = {}

        if vectorizer is None:
            pth = self.get_full_path(self._settings.vectorizer)
            vectorizer = joblib.load(pth)

        self._vec = vectorizer

        self.__load_available_models()

    def load_model(self, model: ModelSettings):
        pth = self.get_full_path(model.name)

        if model.type == "spacy":
            m = SpacyModel(pth)
        elif model.type == "sklearn":
            m = ScikitModel(pth, vectorizer=self._vec)
        else:
            raise ValueError(f"Unrecognized model: {model}")

        self._models[model.name] = m

    def __load_available_models(self):
        total = len(self._settings.available_models)
        print(f"Loading {total} models")
        for model in self._settings.available_models:
            self.load_model(model)

    @property
    def vectorizer(self):
        return self._vec

    def get_full_path(self, name: str):
        return self._settings.model_dir.joinpath(
            f"{name}.{self._settings.model_ext}"
        )

    def get_model(self, model: str) -> Optional[Model]:
        return self._models.get(model, None)
