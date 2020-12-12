import html
import re
from typing import List, Union

import numpy as np
import spacy
from sklearn.base import TransformerMixin
from spacy.language import Doc, Language
from spacy.tokens.token import Token
from spacy.lang.en import STOP_WORDS

try:
    import en_core_web_sm
except ImportError:
    import spacy.cli as spacy_cli

    print("Gotta download the model")
    spacy_cli.download("en_core_web_sm")

nlp: Language = spacy.load(
    "en_core_web_sm", disable=["parser", "tagger", "ner"]
)


class SpacyTokenTransformer(TransformerMixin):
    __symbols = set("!$%^&*()_+|~-=`{}[]:\";'<>?,./-")

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        f = np.vectorize(
            SpacyTokenTransformer.transform_to_tokens, otypes=[np.object]
        )
        X_tokenized = f(X)

        return X_tokenized

    def fit(self, X, y=None, **fit_params):
        return self

    @staticmethod
    def transform_to_tokens(text: np.str) -> List[str]:
        str_text = str(text)
        doc: Doc = nlp(str_text)
        tokens: List[str] = []
        tok: Token
        for tok in doc:
            clean_token: str
            if tok.like_url:
                clean_token = "URL"
            else:
                clean_token = (
                    tok.lemma_.strip().lower()
                )  # if tok.lemma_ != '-PRON-' else tok.lower_
                if (
                    len(clean_token) < 1
                    or clean_token in SpacyTokenTransformer.__symbols
                    or clean_token in STOP_WORDS
                ):
                    continue

            tokens.append(clean_token)

        return tokens


class CleanTextTransformer(TransformerMixin):
    __uplus_pattern = re.compile("\<[uU]\+(?P<digit>[a-zA-Z0-9]+)\>")
    __markup_link_pattern = re.compile("\[(.*)\]\((.*)\)")

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        f = np.vectorize(CleanTextTransformer.transform_clean_text)
        X_clean = f(X)

        return X_clean

    def fit(self, X, y=None, **fit_params):
        return self

    @staticmethod
    def transform_clean_text(raw_text: str):
        try:
            decoded = raw_text.encode("ISO-8859-1").decode("utf-8")
        except:
            decoded = raw_text.encode("ISO-8859-1").decode("cp1252")

        html_unescaped = html.escape(decoded)
        html_unescaped = re.sub(r"\r\n", " ", html_unescaped)
        html_unescaped = re.sub(r"\r\r\n", " ", html_unescaped)
        html_unescaped = re.sub(r"\r", " ", html_unescaped)
        html_unescaped = html_unescaped.replace("&gt;", " > ")
        html_unescaped = html_unescaped.replace("&lt;", " < ")
        html_unescaped = html_unescaped.replace("--", " - ")
        html_unescaped = CleanTextTransformer.__uplus_pattern.sub(
            " U\g<digit>", html_unescaped
        )
        html_unescaped = CleanTextTransformer.__markup_link_pattern.sub(
            " \1 \2", html_unescaped
        )
        html_unescaped = html_unescaped.replace("\\", "")

        return html_unescaped


class TextPreprocessor:
    __slots__ = "_html_cleaner", "_spacy_tokenizer"

    def __init__(self):
        self._html_cleaner = CleanTextTransformer()
        self._spacy_tokenizer = SpacyTokenTransformer()

    def clean_and_tokenize(
        self, txt: Union[str, List[str], np.ndarray]
    ) -> np.ndarray:
        if isinstance(txt, str):
            txt = [txt]

        if isinstance(txt, list):
            txt = np.array(txt, dtype=np.object)

        if not (isinstance(txt, np.ndarray) and txt.dtype == np.object):
            raise ValueError(
                "Input `txt` must be a string, list of strings, or numpy array of type object"
            )

        txt_processed = self._html_cleaner.transform(txt)
        txt_processed = self._spacy_tokenizer.transform(txt_processed)

        return txt_processed

    def clean_and_tokenize_single(self, txt: str) -> List[str]:
        txt_proc = self._html_cleaner.transform_clean_text(txt)
        txt_proc = self._spacy_tokenizer.transform_to_tokens(txt_proc)

        return txt_proc
