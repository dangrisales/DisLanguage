"""
dislanguage.lexical
====================
Extracts lexical-semantic and distributional word-level features from Spanish text.

Backends: spaCy · NLTK WordNet · fastText (gensim) · GPT-2 Spanish (transformers)

Features
--------
f11  Local semantic variability    std cosine dist. to ±3 context words
f12  Global semantic variability   std cosine dist. to all words in text
f13  Familiarity                   psycholinguistic norm (EsPal + Stadthagen)
f14  Imageability                  psycholinguistic norm (EsPal + Stadthagen)
f15  Concreteness                  psycholinguistic norm (EsPal + Stadthagen)
f16  Polysemy                      number of WordNet senses (NLTK omw-1.4)
f17  Corpus frequency              frequency per million (SUBTLEX-ESP)
f18  Within-text frequency         count / total content words (computed)
f19  Word surprisal                -log2 P(word | context) (GPT-2 Spanish)

Features return -1 when the corresponding resource is not loaded.

Requirements
------------
    pip install spacy pandas nltk gensim scipy transformers torch
    python -m spacy download es_core_news_lg
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

    External files (see resources/README.md):
        SUBTLEX-ESP.csv       → f17
        norms_combined.csv    → f13, f14, f15
        cc.es.300.bin (~7GB)  → f11, f12

Usage
-----
    from dislanguage import Lexical

    # Full setup
    ext = Lexical(
        freq_path="resources/SUBTLEX-ESP.csv",
        norms_path="resources/norms_combined.csv",
        vectors_path="resources/cc.es.300.bin",
    )
    ext.load_lm()   # load GPT-2 for f19 (downloads ~500MB on first call)

    df = ext.extract_text("El médico corre rápidamente hacia el paciente.")
    df = ext.extract_file("transcript.txt")
    df.to_csv("features_lexical.csv", index=False)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import pandas as pd
import spacy


# ── Constants ─────────────────────────────────────────────────────────────────

CONTENT_POS = {"VERB", "NOUN", "ADJ", "ADV"}

NOT_FOUND = -1   # sentinel for missing lexicon entries


# ── Module-level spaCy singleton ──────────────────────────────────────────────

_nlp: Optional[spacy.Language] = None


def _get_nlp(model: str = "es_core_news_lg") -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(model)
    return _nlp


# ── Main class ────────────────────────────────────────────────────────────────

class Lexical:
    """
    Lexical-semantic and distributional feature extractor for Spanish text.

    Parameters
    ----------
    freq_path : str, optional
        Path to SUBTLEX-ESP CSV for corpus frequency (f17).
        Expected columns: word, freq_pm
        Download: https://www.bcbl.eu/databases/subtlex-esp/
    norms_path : str, optional
        Path to EsPal / Stadthagen-Gonzalez norms CSV (f13, f14, f15).
        Expected columns: word, familiarity, imageability, concreteness
        Download: http://www.bcbl.eu/databases/espal/
    vectors_path : str, optional
        Path to fastText binary model for semantic variability (f11, f12).
        Expected format: cc.es.300.bin
        Download: https://fasttext.cc/docs/en/crawl-vectors.html
    lm_name : str, optional
        HuggingFace model name for surprisal (f19).
        Default: 'datificate/gpt2-small-spanish'
    model : str
        spaCy model name. Default: 'es_core_news_lg'.

    Examples
    --------
    >>> ext = Lexical()
    >>> df  = ext.extract_text("El médico corre rápidamente hacia el paciente.")
    >>> print(df[["form", "f16_polysemy", "f17_corpus_freq", "f18_text_freq"]])
    """

    def __init__(
        self,
        freq_path:    Optional[str] = None,
        norms_path:   Optional[str] = None,
        vectors_path: Optional[str] = None,
        word_vectors                = None,   # pre-loaded gensim KeyedVectors
        lm_name:      str           = "datificate/gpt2-small-spanish",
        model:        str           = "es_core_news_lg",
    ):
        self._model   = model
        self._lm_name = lm_name
        self._lm      = None
        self._lm_tok  = None

        # Load optional resources
        self._freq  = self._load_freq(freq_path)   if freq_path   else {}
        self._norms = self._load_norms(norms_path) if norms_path  else {}

        # Accept pre-loaded vectors (shared with Semantic) or load from path
        if word_vectors is not None:
            self._wv = word_vectors
        elif vectors_path:
            self._wv = self._load_vectors(vectors_path)
        else:
            self._wv = None

    # ── Public API ────────────────────────────────────────────

    def extract_text(self, text: str) -> pd.DataFrame:
        """
        Extract lexical features from a plain-text string.

        Parameters
        ----------
        text : str
            Raw Spanish text (one or more sentences).

        Returns
        -------
        pd.DataFrame
            One row per content word (VERB / NOUN / ADJ / ADV).
            Columns: form, lemma, pos, f11…f19.
            Stubs return -1 until the corresponding resource is loaded.
        """
        nlp = _get_nlp(self._model)
        doc = nlp(text)

        # Collect all content word tokens with their index
        content_tokens = [(i, t) for i, t in enumerate(doc)
                          if t.pos_ in CONTENT_POS]

        # Lemma sequence of content words only (for context windows)
        lemma_seq = [t.lemma_.lower() for _, t in content_tokens]

        # Within-text frequency over all tokens (f18)
        all_lemmas  = [t.lemma_.lower() for t in doc if t.pos_ in CONTENT_POS]
        text_counts = Counter(all_lemmas)
        total       = len(all_lemmas)

        records = []
        for seq_idx, (_, token) in enumerate(content_tokens):
            records.append(
                self._extract_token(token, seq_idx, lemma_seq, text_counts, total)
            )

        return pd.DataFrame(records)

    def extract_file(self, path: str, encoding: str = "utf-8") -> pd.DataFrame:
        """
        Extract lexical features from a plain-text .txt file.

        Parameters
        ----------
        path : str
            Path to the input file. Plain UTF-8 text.

        Returns
        -------
        pd.DataFrame
            Same format as extract_text().
        """
        text = open(path, encoding=encoding).read().strip()
        return self.extract_text(text)

    # ── Token-level extraction ────────────────────────────────

    def _extract_token(
        self,
        token:       spacy.tokens.Token,
        seq_idx:     int,
        lemma_seq:   list[str],
        text_counts: Counter,
        total:       int,
    ) -> dict:
        form  = token.text
        lemma = token.lemma_.lower()
        pos   = token.pos_

        # Context windows (±3 content words)
        left3  = lemma_seq[max(0, seq_idx - 3): seq_idx]
        right3 = lemma_seq[seq_idx + 1: seq_idx + 4]
        ctx    = left3 + right3

        return {
            # ── Metadata
            "form":  form,
            "lemma": lemma,
            "pos":   pos,

            # ── f11  Local semantic variability
            # std of cosine distances to ±3 context words
            # Stub — activate by passing vectors_path to Lexical()
            # Setup: pip install gensim scipy
            #        Download cc.es.300.bin from fasttext.cc (~7GB)
            "f11_local_sem_var": self._local_sem_var(lemma, ctx),

            # ── f12  Global semantic variability
            # std of cosine distances to all content words in text
            # Stub — same setup as f11
            "f12_global_sem_var": self._global_sem_var(lemma, lemma_seq),

            # ── f13  Familiarity
            # Stub — activate by passing norms_path to Lexical()
            # Setup: download EsPal from bcbl.eu/databases/espal/
            #        CSV columns: word, familiarity, imageability, concreteness
            "f13_familiarity": self._norms.get(lemma, {}).get("familiarity", NOT_FOUND),

            # ── f14  Imageability
            # Stub — same setup as f13
            "f14_imageability": self._norms.get(lemma, {}).get("imageability", NOT_FOUND),

            # ── f15  Concreteness
            # Stub — same setup as f13
            "f15_concreteness": self._norms.get(lemma, {}).get("concreteness", NOT_FOUND),

            # ── f16  Polysemy — number of WordNet senses (NLTK)
            # Setup: pip install nltk
            #        python -c "import nltk; nltk.download('wordnet');
            #                   nltk.download('omw-1.4')"
            "f16_polysemy": self._polysemy(lemma, pos),

            # ── f17  Corpus frequency (per million)
            # Stub — activate by passing freq_path to Lexical()
            # Setup: download SUBTLEX-ESP from bcbl.eu/databases/subtlex-esp/
            #        CSV columns: word, freq_pm
            "f17_corpus_freq": self._freq.get(lemma, NOT_FOUND),

            # ── f18  Within-text frequency (normalised over content words)
            # Fully computed — no external resource needed
            "f18_text_freq": round(text_counts.get(lemma, 0) / total, 6)
                             if total > 0 else 0,

            # ── f19  Word surprisal  (-log2 P(word | context))
            # Stub — activate by calling ext.load_lm()
            # Setup: pip install transformers torch
            #        Model: datificate/gpt2-small-spanish (~500MB, auto-downloaded)
            "f19_surprisal": self._surprisal(form, [
                t for t in token.doc[:token.i]
            ]),
        }

    # ── Feature implementations ───────────────────────────────

    def _local_sem_var(self, lemma: str, context: list[str]) -> float:
        if self._wv is None:
            return NOT_FOUND
        try:
            from scipy.spatial.distance import cosine
            import numpy as np
            if lemma not in self._wv:
                return NOT_FOUND
            dists = [cosine(self._wv[lemma], self._wv[w])
                     for w in context if w in self._wv]
            return round(float(np.std(dists)), 6) if len(dists) > 1 else NOT_FOUND
        except Exception:
            return NOT_FOUND

    def _global_sem_var(self, lemma: str, all_lemmas: list[str]) -> float:
        if self._wv is None:
            return NOT_FOUND
        try:
            from scipy.spatial.distance import cosine
            import numpy as np
            if lemma not in self._wv:
                return NOT_FOUND
            others = [w for w in all_lemmas if w != lemma and w in self._wv]
            dists  = [cosine(self._wv[lemma], self._wv[w]) for w in others]
            return round(float(np.std(dists)), 6) if len(dists) > 1 else NOT_FOUND
        except Exception:
            return NOT_FOUND

    def _polysemy(self, lemma: str, pos: str) -> int:
        try:
            from nltk.corpus import wordnet as wn
            pos_wn = {
                "NOUN": wn.NOUN,
                "VERB": wn.VERB,
                "ADJ":  wn.ADJ,
                "ADV":  wn.ADV,
            }.get(pos)
            if pos_wn is None:
                return NOT_FOUND
            synsets = wn.synsets(lemma, pos=pos_wn, lang="spa")
            return len(synsets) if synsets else NOT_FOUND
        except Exception:
            return NOT_FOUND

    def _surprisal(self, form: str, left_tokens) -> float:
        if self._lm is None:
            return NOT_FOUND
        try:
            import torch
            context = " ".join([t.text for t in left_tokens]) + " " + form
            inputs  = self._lm_tok(context, return_tensors="pt")
            with torch.no_grad():
                logits = self._lm(**inputs).logits
            log_probs = torch.log_softmax(logits[0, -2], dim=-1)
            word_ids  = self._lm_tok(form, add_special_tokens=False)["input_ids"]
            if not word_ids:
                return NOT_FOUND
            return round(float(-log_probs[word_ids[0]].item() / math.log(2)), 6)
        except Exception:
            return NOT_FOUND

    def load_lm(self) -> None:
        """
        Explicitly load the language model for surprisal (f19).
        Call this once before processing if you want f19 values.

        Setup:
            pip install transformers torch

        Example:
            ext = Lexical()
            ext.load_lm()
            df = ext.extract_text("El médico corre.")
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print(f"  Loading language model '{self._lm_name}'...")
            self._lm_tok = AutoTokenizer.from_pretrained(self._lm_name)
            self._lm     = AutoModelForCausalLM.from_pretrained(self._lm_name)
            self._lm.eval()
            print("  Language model loaded.")
        except ImportError:
            raise ImportError(
                "transformers and torch are required for surprisal (f19).\n"
                "Install with:  pip install transformers torch"
            )

    # ── Resource loaders ─────────────────────────────────────

    @staticmethod
    def _load_freq(path: str) -> dict:
        try:
            df = pd.read_csv(path)
            print(f"  Frequency lexicon loaded: {len(df):,} entries from '{path}'")
            return dict(zip(df["word"].str.lower(), df["freq_pm"].astype(float)))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Frequency file not found: '{path}'\n"
                "Download SUBTLEX-ESP from:\n"
                "  https://www.bcbl.eu/databases/subtlex-esp/\n"
                "Expected CSV columns: word, freq_pm"
            )

    @staticmethod
    def _load_norms(path: str) -> dict:
        try:
            df    = pd.read_csv(path)
            norms = {}
            for _, row in df.iterrows():
                norms[str(row["word"]).lower()] = {
                    "familiarity":  float(row.get("familiarity",  NOT_FOUND) or NOT_FOUND),
                    "imageability": float(row.get("imageability", NOT_FOUND) or NOT_FOUND),
                    "concreteness": float(row.get("concreteness", NOT_FOUND) or NOT_FOUND),
                }
            print(f"  Norms lexicon loaded: {len(norms):,} entries from '{path}'")
            return norms
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Norms file not found: '{path}'\n"
                "Download EsPal from:\n"
                "  http://www.bcbl.eu/databases/espal/\n"
                "Expected CSV columns: word, familiarity, imageability, concreteness"
            )

    @staticmethod
    def _load_vectors(path: str):
        """
        Load word vectors from any supported format.
        .bin   → fastText (cc.es.300.bin) or word2vec binary (word2vec_es.bin)
        .vec   → word2vec text format
        .model → gensim native Word2Vec format
        """
        try:
            from gensim.models import KeyedVectors, Word2Vec
            from gensim.models.fasttext import load_facebook_model

            print(f"  Loading word vectors from '{path}' (this may take a while)...")

            if "fasttext" in path.lower() or "cc.es" in path.lower():
                wv  = load_facebook_model(path).wv
                fmt = "fastText"
            elif path.endswith(".model"):
                wv  = Word2Vec.load(path).wv
                fmt = "gensim Word2Vec"
            elif path.endswith((".vec", ".txt")):
                wv  = KeyedVectors.load_word2vec_format(path, binary=False)
                fmt = "word2vec text"
            else:
                # Default .bin → word2vec binary
                wv  = KeyedVectors.load_word2vec_format(path, binary=True)
                fmt = "word2vec binary"

            print(f"  Word vectors loaded: {len(wv):,} entries  [{fmt}]")
            return wv

        except ImportError:
            raise ImportError(
                "gensim is required for vector features.\n"
                "Install with:  pip install gensim scipy"
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Vectors file not found: '{path}'")