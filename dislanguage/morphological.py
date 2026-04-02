"""
dislanguage.morphological
==========================
Extracts morphological and phonological word-level features from Spanish text.

Backends: spaCy (es_core_news_lg) · pyphen · phonemizer · Morfessor

Features
--------
f01  Content-word type     0=VERB, 1=NOUN, 2=ADJ, 3=ADV
f02  N consonants          counted from token surface form
f03  N vowels              counted from token surface form
f04  N characters          len(form)
f05  N phonemes            phonemizer + espeak-ng  (None if not installed)
f06  N syllables           pyphen Spanish dictionary
f07  Verbal tense          0=Pres, 1=Past, 2=Imp, 3=Fut, -1=N/A
f08  Person                1, 2, 3, -1=N/A
f09  Grammatical number    0=Sing, 1=Plur, -1=N/A
f10  N morphemes           Morfessor model  (None if model not provided)

Requirements
------------
    pip install spacy pyphen pandas phonemizer morfessor
    sudo apt install espeak-ng
    python -m spacy download es_core_news_lg

    # Train Morfessor model once (see resources/README.md):
    #   python scripts/train_morfessor.py

Usage
-----
    from dislanguage import Morphological

    # Basic — f05 needs phonemizer+espeak-ng, f10 needs morfessor_path
    ext = Morphological()
    df  = ext.extract_text("El médico corre rápidamente.")

    # Full — all features active
    ext = Morphological(morfessor_path="resources/es_morfessor.bin")
    df  = ext.extract_text("El médico corre rápidamente.")
    df  = ext.extract_file("transcript.txt")
    df.to_csv("features_morphological.csv", index=False)
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
import pyphen
import spacy


# ── Constants ─────────────────────────────────────────────────────────────────

VOWELS       = set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")
CONSONANT_RE = re.compile(r"[bcdfghjklmnñpqrstvwxyzBCDFGHJKLMNÑPQRSTVWXYZ]")

CONTENT_POS = {"VERB", "NOUN", "ADJ", "ADV"}

POS_CODE = {"VERB": 0, "NOUN": 1, "ADJ": 2, "ADV": 3}

TENSE_CODE  = {"Pres": 0, "Past": 1, "Imp": 2, "Fut": 3}
PERSON_CODE = {"1": 1, "2": 2, "3": 3}
NUMBER_CODE = {"Sing": 0, "Plur": 1}


# ── Module-level singletons (loaded once) ─────────────────────────────────────

_nlp:    Optional[spacy.Language] = None
_pyphen: pyphen.Pyphen             = pyphen.Pyphen(lang="es")

# phonemizer is optional — checked lazily on first use
_phonemizer_ok: Optional[bool] = None


def _has_phonemizer() -> bool:
    global _phonemizer_ok
    if _phonemizer_ok is None:
        try:
            from phonemizer import phonemize  # noqa: F401
            _phonemizer_ok = True
        except ImportError:
            _phonemizer_ok = False
    return _phonemizer_ok


def _get_nlp(model: str = "es_core_news_lg") -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(model)
    return _nlp


# ── Main class ────────────────────────────────────────────────────────────────

class Morphological:
    """
    Morphological and phonological feature extractor for Spanish text.

    Parameters
    ----------
    model : str
        spaCy model name. Default: 'es_core_news_lg'.

    Examples
    --------
    >>> ext = Morphological()
    >>> df  = ext.extract_text("El médico corre hacia el paciente.")
    >>> print(df[["form", "f01_pos_label", "f06_n_syllables", "f07_verbal_tense"]])
          form f01_pos_label  f06_n_syllables f07_verbal_tense
    0   médico          NOUN                3             None
    1    corre          VERB                2             Pres
    2  paciente          NOUN                4             None
    """

    def __init__(
        self,
        model:          str           = "es_core_news_lg",
        morfessor_path: Optional[str] = None,
    ):
        self._model     = model
        self._morfessor = None

        if morfessor_path:
            try:
                import morfessor as _morf
                io = _morf.MorfessorIO()
                self._morfessor = io.read_binary_model_file(morfessor_path)
                print(f"  Morfessor model loaded from '{morfessor_path}'")
            except ImportError:
                raise ImportError(
                    "morfessor is required for f10.\n"
                    "Install with:  pip install morfessor"
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Morfessor model not found: '{morfessor_path}'\n"
                    "Train a model first — see README for instructions."
                )

    # ── Public API ────────────────────────────────────────────

    def extract_text(self, text: str) -> pd.DataFrame:
        """
        Extract features from a plain-text string.

        Parameters
        ----------
        text : str
            Raw Spanish text (one or more sentences).

        Returns
        -------
        pd.DataFrame
            One row per content word (VERB / NOUN / ADJ / ADV).
            Columns: form, lemma, pos, f01…f10.
        """
        nlp = _get_nlp(self._model)
        doc = nlp(text)
        records = [self._extract_token(token) for token in doc
                   if token.pos_ in CONTENT_POS]
        return pd.DataFrame(records)

    def extract_file(self, path: str, encoding: str = "utf-8") -> pd.DataFrame:
        """
        Extract features from a plain-text .txt file.

        Parameters
        ----------
        path : str
            Path to the input file. One or more sentences, plain UTF-8 text.
        encoding : str
            File encoding (default 'utf-8').

        Returns
        -------
        pd.DataFrame
            Same format as extract_text().
        """
        text = open(path, encoding=encoding).read().strip()
        return self.extract_text(text)

    # ── Token-level extraction ────────────────────────────────

    def _extract_token(self, token: spacy.tokens.Token) -> dict:
        """Build the feature dict for a single content-word token."""
        form  = token.text
        lemma = token.lemma_.lower()
        lemma = lemma.split()[0]
        pos   = token.pos_
        morph = token.morph.to_dict()

        return {
            # ── Metadata
            "form":  form,
            "lemma": lemma,
            "pos":   pos,

            # ── f01  Content-word type
            "f01_pos_code":  POS_CODE[pos],
            "f01_pos_label": pos,

            # ── f02  Consonant count
            "f02_n_consonants": self._count_consonants(form),

            # ── f03  Vowel count
            "f03_n_vowels": self._count_vowels(form),

            # ── f04  Character count
            "f04_n_chars": len(form),

            # ── f05  Phoneme count  (phonemizer + espeak-ng)
            "f05_n_phonemes": self._count_phonemes(form),

            # ── f06  Syllable count  (pyphen)
            "f06_n_syllables": self._count_syllables(form),

            # ── f07  Verbal tense  (0=Pres, 1=Past, 2=Imp, 3=Fut, -1=N/A)
            "f07_verbal_tense": TENSE_CODE.get(morph.get("Tense"), -1),

            # ── f08  Person  (1, 2, 3, -1=N/A)
            "f08_person": PERSON_CODE.get(morph.get("Person"), -1),

            # ── f09  Grammatical number  (0=Sing, 1=Plur, -1=N/A)
            "f09_number": NUMBER_CODE.get(morph.get("Number"), -1),

            # ── f10  Morpheme count  (Morfessor)
            "f10_n_morphemes": self._count_morphemes(lemma),
        }

    # ── String-level helpers ──────────────────────────────────

    def _count_morphemes(self, lemma: str) -> Optional[int]:
        """Count morphemes using the trained Morfessor model."""
        if self._morfessor is None:
            return None
        result = self._morfessor.viterbi_segment(lemma)
        return len(result[0])

    @staticmethod
    def _count_phonemes(word: str) -> Optional[int]:
        if not _has_phonemizer():
            return None
        from phonemizer import phonemize
        phones = phonemize(word, language="es", backend="espeak", with_stress=False)
        return len(phones.replace(" ", ""))

    @staticmethod
    def _count_vowels(word: str) -> int:
        return sum(1 for c in word if c in VOWELS)

    @staticmethod
    def _count_consonants(word: str) -> int:
        return len(CONSONANT_RE.findall(word))

    @staticmethod
    def _count_syllables(word: str) -> int:
        result = _pyphen.inserted(word.lower())
        return (result.count("-") + 1) if result else 1