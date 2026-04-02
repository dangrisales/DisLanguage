"""
dislanguage.discourse
======================
Extracts text-level (discourse) features from Spanish text.

Unlike the other extractors which return one row per content word,
this extractor returns ONE ROW PER TEXT — a single feature vector
describing the whole transcript.

Backends: spaCy (es_core_news_lg)

Features
--------
POS ratios relative to content words
--------------------------------------
r_verb_content    VERB / N content words
r_noun_content    NOUN / N content words
r_adj_content     ADJ  / N content words
r_adv_content     ADV  / N content words

POS ratios relative to total tokens
--------------------------------------
r_verb_total      VERB / N total tokens
r_noun_total      NOUN / N total tokens
r_adj_total       ADJ  / N total tokens
r_adv_total       ADV  / N total tokens

Lexical counts
--------------
n_total_tokens    Total number of tokens (excl. spaces)
n_content_words   Total number of content words
n_sentences       Number of sentences
content_ratio     N content words / N total tokens

Lexical diversity
-----------------
type_token_ratio  Unique lemmas / N content words (TTR)
                  High TTR → more varied vocabulary
                  Low TTR  → more repetitive vocabulary

Syntactic
---------
mean_sent_length  Mean tokens per sentence
std_sent_length   Std of tokens per sentence

Requirements
------------
    pip install spacy pandas
    python -m spacy download es_core_news_lg

Usage
-----
    from dislanguage import Discourse

    ext = Discourse()

    # From a string — returns a single-row DataFrame
    df = ext.extract_text("El médico corre rápidamente hacia el paciente.")

    # From a .txt file
    df = ext.extract_file("transcript.txt")

    # Process multiple subjects
    import pandas as pd
    from pathlib import Path

    ext     = Discourse()
    results = []
    for txt_file in Path("data/transcriptions/").glob("*.txt"):
        df = ext.extract_file(str(txt_file))
        df.insert(0, "subject_id", txt_file.stem)
        results.append(df)

    all_df = pd.concat(results, ignore_index=True)
    all_df.to_csv("features_discourse.csv", index=False)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import spacy


# ── Constants ─────────────────────────────────────────────────────────────────

CONTENT_POS = {"VERB", "NOUN", "ADJ", "ADV"}


# ── Module-level spaCy singleton ──────────────────────────────────────────────

_nlp: Optional[spacy.Language] = None


def _get_nlp(model: str = "es_core_news_lg") -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(model)
    return _nlp


# ── Main class ────────────────────────────────────────────────────────────────

class Discourse:
    """
    Text-level (discourse) feature extractor for Spanish text.

    Returns ONE ROW PER TEXT with all features computed over the
    full transcript, unlike the word-level extractors.

    Parameters
    ----------
    model : str
        spaCy model name. Default: 'es_core_news_lg'.

    Examples
    --------
    >>> ext = Discourse()
    >>> df  = ext.extract_text("El médico corre rápidamente hacia el paciente.")
    >>> print(df[["r_verb_content", "r_noun_content", "type_token_ratio",
    ...           "mean_sent_length"]].T)
    """

    def __init__(self, model: str = "es_core_news_lg"):
        self._model = model

    # ── Public API ────────────────────────────────────────────

    def extract_text(self, text: str) -> pd.DataFrame:
        """
        Extract discourse features from a plain-text string.

        Parameters
        ----------
        text : str
            Raw Spanish text (one or more sentences).

        Returns
        -------
        pd.DataFrame
            Single row with all discourse features.
        """
        nlp = _get_nlp(self._model)
        doc = nlp(text)
        return pd.DataFrame([self._extract_doc(doc)])

    def extract_file(self, path: str, encoding: str = "utf-8") -> pd.DataFrame:
        """
        Extract discourse features from a plain-text .txt file.

        Parameters
        ----------
        path : str
            Path to the input file. Plain UTF-8 text.
        encoding : str
            File encoding (default 'utf-8').

        Returns
        -------
        pd.DataFrame
            Single row with all discourse features.
        """
        text = open(path, encoding=encoding).read().strip()
        return self.extract_text(text)

    # ── Document-level extraction ─────────────────────────────

    def _extract_doc(self, doc: spacy.tokens.Doc) -> dict:
        """Compute all discourse features for a spaCy Doc."""

        # ── Token counts ──────────────────────────────────────
        all_tokens     = [t for t in doc if not t.is_space]
        content_tokens = [t for t in all_tokens if t.pos_ in CONTENT_POS]

        n_total    = len(all_tokens)
        n_content  = len(content_tokens)

        # ── POS counts ────────────────────────────────────────
        pos_counts = {"VERB": 0, "NOUN": 0, "ADJ": 0, "ADV": 0}
        for t in content_tokens:
            pos_counts[t.pos_] += 1

        def _rc(pos: str) -> float:
            """Ratio relative to content words."""
            return round(pos_counts[pos] / n_content, 6) if n_content > 0 else 0.0

        def _rt(pos: str) -> float:
            """Ratio relative to total tokens."""
            return round(pos_counts[pos] / n_total, 6) if n_total > 0 else 0.0

        # ── Lexical diversity (TTR) ───────────────────────────
        # Type-Token Ratio: unique lemmas / total content words
        # Sensitive to text length — for short texts (< 100 words) it is
        # a reasonable measure; for longer texts consider MATTR or MTLD.
        unique_lemmas = len({t.lemma_.lower() for t in content_tokens})
        ttr = round(unique_lemmas / n_content, 6) if n_content > 0 else 0.0

        # ── Sentence-level statistics ─────────────────────────
        sentences    = list(doc.sents)
        n_sentences  = len(sentences)
        sent_lengths = [
            len([t for t in s if not t.is_space])
            for s in sentences
        ]

        if sent_lengths:
            mean_sl = round(sum(sent_lengths) / len(sent_lengths), 4)
            # population std
            variance = sum((l - mean_sl) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_sl   = round(variance ** 0.5, 4)
        else:
            mean_sl = 0.0
            std_sl  = 0.0

        # ── Assemble feature dict ─────────────────────────────
        return {
            # ── POS ratios / content words
            "r_verb_content": _rc("VERB"),
            "r_noun_content": _rc("NOUN"),
            "r_adj_content":  _rc("ADJ"),
            "r_adv_content":  _rc("ADV"),

            # ── POS ratios / total tokens
            "r_verb_total":   _rt("VERB"),
            "r_noun_total":   _rt("NOUN"),
            "r_adj_total":    _rt("ADJ"),
            "r_adv_total":    _rt("ADV"),

            # ── Lexical counts
            "n_total_tokens":   n_total,
            "n_content_words":  n_content,
            "n_sentences":      n_sentences,
            "content_ratio":    round(n_content / n_total, 6) if n_total > 0 else 0.0,

            # ── Lexical diversity
            "type_token_ratio": ttr,

            # ── Syntactic
            "mean_sent_length": mean_sl,
            "std_sent_length":  std_sl,
        }