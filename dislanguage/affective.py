"""
dislanguage.affective
======================
Extracts emotional and affective word-level features from Spanish text.

Backend: spaCy (es_core_news_lg) + NRC-VAD Spanish lexicon.

Features
--------
f25  Emotion polarity         sign(valence) → -1=negative, 0=neutral, 1=positive
f26  Valence magnitude        |valence|
f27  Arousal magnitude        |arousal|
f28  Emotional activation     |valence| x |arousal|
f29  Dominance                raw dominance score
f30  Distance to neutral      sqrt(valence² + arousal²)

Words not found in the VAD lexicon return -1 for all features.

Requirements
------------
    pip install spacy pandas
    python -m spacy download es_core_news_lg

    External files (see resources/README.md):
        NRC-VAD-es.csv   → f25-f30
        Download: https://saifmohammad.com/WebPages/nrc-vad.html
        Columns: word, valence, arousal, dominance

Usage
-----
    from dislanguage import Affective

    ext = Affective(vad_path="resources/NRC-VAD-es.csv")

    df = ext.extract_text("El paciente camina lentamente.")
    df = ext.extract_file("transcript.txt")
    df.to_csv("features_affective.csv", index=False)

    # Check lexicon coverage on your corpus
    print(ext.coverage("El paciente camina lentamente."))

    df.to_csv("features_affective.csv", index=False)
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd
import spacy


# ── Constants ─────────────────────────────────────────────────────────────────

CONTENT_POS = {"VERB", "NOUN", "ADJ", "ADV"}

POS_CODE = {"VERB": 0, "NOUN": 1, "ADJ": 2, "ADV": 3}

# Sentinel for words not found in the VAD lexicon
NOT_FOUND = -1


# ── Module-level spaCy singleton ──────────────────────────────────────────────

_nlp: Optional[spacy.Language] = None


def _get_nlp(model: str = "es_core_news_lg") -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(model)
    return _nlp


# ── Main class ────────────────────────────────────────────────────────────────

class Affective:
    """
    Emotional and affective feature extractor for Spanish text.

    Parameters
    ----------
    vad_path : str
        Path to the VAD lexicon CSV file.
        Required columns: word, valence, arousal, dominance
        Recommended: NRC-VAD Spanish translation or ANEW-ES.
    word_col : str
        Name of the word column in the CSV. Default: 'word'.
    valence_col : str
        Name of the valence column. Default: 'valence'.
    arousal_col : str
        Name of the arousal column. Default: 'arousal'.
    dominance_col : str
        Name of the dominance column. Default: 'dominance'.
    model : str
        spaCy model name. Default: 'es_core_news_lg'.

    Examples
    --------
    >>> ext = Affective(vad_path="NRC-VAD-es.csv")
    >>> df  = ext.extract_text("El paciente camina lentamente.")
    >>> print(df[["form", "f25_polarity", "f26_valence_mag",
    ...           "f27_arousal_mag", "f30_dist_neutral"]])
    """

    def __init__(
        self,
        vad_path: str,
        word_col:      str = "word",
        valence_col:   str = "valence",
        arousal_col:   str = "arousal",
        dominance_col: str = "dominance",
        model:         str = "es_core_news_lg",
    ):
        self._model = model
        self._vad   = self._load_vad(
            vad_path, word_col, valence_col, arousal_col, dominance_col
        )

    # ── Public API ────────────────────────────────────────────

    def extract_text(self, text: str) -> pd.DataFrame:
        """
        Extract affective features from a plain-text string.

        Parameters
        ----------
        text : str
            Raw Spanish text (one or more sentences).

        Returns
        -------
        pd.DataFrame
            One row per content word (VERB / NOUN / ADJ / ADV).
            Columns: form, lemma, pos, f25…f30.
            Words not found in the VAD lexicon return -1.
        """
        nlp = _get_nlp(self._model)
        doc = nlp(text)
        records = [self._extract_token(token) for token in doc
                   if token.pos_ in CONTENT_POS]
        return pd.DataFrame(records)

    def extract_file(self, path: str, encoding: str = "utf-8") -> pd.DataFrame:
        """
        Extract affective features from a plain-text .txt file.

        Parameters
        ----------
        path : str
            Path to the input file. Plain UTF-8 text.
        encoding : str
            File encoding (default 'utf-8').

        Returns
        -------
        pd.DataFrame
            Same format as extract_text().
        """
        text = open(path, encoding=encoding).read().strip()
        return self.extract_text(text)

    def coverage(self, text: str) -> float:
        """
        Return the fraction of content words found in the VAD lexicon.
        Useful to assess how well the lexicon covers your corpus.

        Parameters
        ----------
        text : str
            Raw Spanish text.

        Returns
        -------
        float
            Value between 0.0 and 1.0.
        """
        nlp = _get_nlp(self._model)
        doc = nlp(text)
        content = [t for t in doc if t.pos_ in CONTENT_POS]
        if not content:
            return 0.0
        found = sum(1 for t in content if t.lemma_.lower() in self._vad)
        return found / len(content)

    # ── Token-level extraction ────────────────────────────────

    def _extract_token(self, token: spacy.tokens.Token) -> dict:
        """Build the affective feature dict for a single content-word token."""
        form  = token.text
        lemma = token.lemma_.lower()
        pos   = token.pos_

        v, a, d = self._lookup(lemma)

        return {
            # ── Metadata
            "form":  form,
            "lemma": lemma,
            "pos":   pos,

            # ── f25  Emotion polarity  (-1=negative, 0=neutral, 1=positive, -1=N/A)
            "f25_polarity": self._sign(v) if v != NOT_FOUND else NOT_FOUND,

            # ── f26  Valence magnitude  (|valence|, -1=N/A)
            "f26_valence_mag": round(abs(v), 4) if v != NOT_FOUND else NOT_FOUND,

            # ── f27  Arousal magnitude  (|arousal|, -1=N/A)
            "f27_arousal_mag": round(abs(a), 4) if a != NOT_FOUND else NOT_FOUND,

            # ── f28  Emotional activation  (|v| x |a|, -1=N/A)
            "f28_emotional_activation": round(abs(v) * abs(a), 4)
                                        if v != NOT_FOUND and a != NOT_FOUND
                                        else NOT_FOUND,

            # ── f29  Dominance  (raw score, -1=N/A)
            "f29_dominance": round(d, 4) if d != NOT_FOUND else NOT_FOUND,

            # ── f30  Distance to neutral  (√(v²+a²), -1=N/A)
            "f30_dist_neutral": round(math.sqrt(v**2 + a**2), 4)
                                if v != NOT_FOUND and a != NOT_FOUND
                                else NOT_FOUND,
        }

    # ── VAD lookup ────────────────────────────────────────────

    def _lookup(self, lemma: str) -> tuple[float, float, float]:
        """
        Look up valence, arousal, dominance for a lemma.
        Returns (NOT_FOUND, NOT_FOUND, NOT_FOUND) if the word is missing.
        """
        entry = self._vad.get(lemma)
        if entry is None:
            return NOT_FOUND, NOT_FOUND, NOT_FOUND
        return entry["valence"], entry["arousal"], entry["dominance"]

    # ── Static helpers ────────────────────────────────────────

    @staticmethod
    def _sign(x: float) -> int:
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    # ── Resource loader ───────────────────────────────────────

    @staticmethod
    def _load_vad(
        path: str,
        word_col:      str,
        valence_col:   str,
        arousal_col:   str,
        dominance_col: str,
    ) -> dict[str, dict]:
        """
        Load a VAD lexicon from a CSV file into a dict keyed by lowercase word.
        """
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"VAD lexicon not found: '{path}'\n"
                "Download the NRC-VAD lexicon from:\n"
                "  https://saifmohammad.com/WebPages/nrc-vad.html\n"
                "Format it as a CSV with columns: word, valence, arousal, dominance"
            )

        missing = [c for c in [word_col, valence_col, arousal_col, dominance_col]
                   if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns in VAD file: {missing}\n"
                f"Found columns: {list(df.columns)}\n"
                "Use the word_col/valence_col/arousal_col/dominance_col parameters "
                "to map your column names."
            )

        vad = {}
        for _, row in df.iterrows():
            vad[str(row[word_col]).lower()] = {
                "valence":   float(row[valence_col]),
                "arousal":   float(row[arousal_col]),
                "dominance": float(row[dominance_col]),
            }

        print(f"  VAD lexicon loaded: {len(vad):,} entries from '{path}'")
        return vad