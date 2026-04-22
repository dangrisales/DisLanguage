"""
dislanguage.semantic
=====================
Extracts action-related semantic word-level features from Spanish text.

Backends: spaCy · fastText / Word2Vec (gensim) · motility norms · NRC-VAD · norms_combined

Features
--------
f20  Motor content              motility score (0.0 if word not in corpus)
f21  Distance to manipulation   cosine dist. to manipulable-objects centroid
f22  Distance to motor action   cosine dist. to motor-action centroid
f23  Distance to abstract cog.  cosine dist. to abstract-cognition centroid
f24  Emotion x motor            valence x motility (-1 if valence missing)

Features return -1 when the corresponding resource is not loaded.
f21 and f23 require norms_path — they will not fall back to default word lists.

Requirements
------------
    pip install spacy pandas gensim scipy
    python -m spacy download es_core_news_lg

    External files (see resources/README.md):
        motility_scores.csv   → f20, f24
        NRC-VAD-es.csv        → f24
        norms_combined.csv    → f21, f23  (concreteness norms)
        cc.es.300.bin (~7GB)  → f21, f22, f23  (or word2vec_es.bin)

Usage
-----
    from dislanguage import Semantic

    ext = Semantic(
        motility_path="resources/motility_scores.csv",
        vad_path="resources/NRC-VAD-es.csv",
        vectors_path="resources/cc.es.300.bin",
        norms_path="resources/norms_combined.csv",
        manipulation_threshold=6.0,
        abstract_threshold=3.0,
    )

    df = ext.extract_text("El médico corre rápidamente hacia el paciente.")
    df.to_csv("features_semantic.csv", index=False)

    print(ext.coverage("El médico corre rápidamente hacia el paciente."))
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import spacy


# ── Constants ─────────────────────────────────────────────────────────────────

CONTENT_POS = {"VERB", "NOUN", "ADJ", "ADV"}
NOT_FOUND   = -1

DEFAULT_MOTOR_WORDS = [
    "correr", "saltar", "caminar", "lanzar", "agarrar",
    "empujar", "golpear", "patear", "trepar", "nadar",
    "girar", "levantar", "tirar", "arrastrar", "señalar",
    "coger", "soltar", "doblar", "estirar", "mover",
]


# ── spaCy singleton ───────────────────────────────────────────────────────────

_nlp: Optional[spacy.Language] = None


def _get_nlp(model: str = "es_core_news_lg") -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(model)
    return _nlp


# ── Main class ────────────────────────────────────────────────────────────────

class Semantic:
    """
    Action-related semantic feature extractor for Spanish text.

    Parameters
    ----------
    motility_path : str, optional
        Path to motor content norms CSV (word, score).
        Required for f20 and f24.
    vad_path : str, optional
        Path to NRC-VAD Spanish lexicon CSV (word, valence, arousal, dominance).
        Required for f24.
    vectors_path : str, optional
        Path to word vector file (.bin fastText or word2vec, .vec, .model).
        Required for f21, f22, f23.
    word_vectors : KeyedVectors, optional
        Pre-loaded gensim KeyedVectors — shared with Lexical to avoid
        loading the same file twice.
    norms_path : str, optional
        Path to psycholinguistic norms CSV (word, concreteness, ...).
        Required for f21 (manipulation cluster) and f23 (abstract cluster).
        If not provided, f21 and f23 return -1.
    manipulation_threshold : float
        Minimum concreteness score to include a word in the manipulation
        cluster centroid for f21. Default: 6.0 (scale 1-7).
    abstract_threshold : float
        Maximum concreteness score to include a word in the abstract
        cognition cluster centroid for f23. Default: 3.0 (scale 1-7).
    model : str
        spaCy model name. Default: 'es_core_news_lg'.
    """

    def __init__(
        self,
        motility_path:          Optional[str]   = None,
        vad_path:               Optional[str]   = None,
        vectors_path:           Optional[str]   = None,
        word_vectors                            = None,
        norms_path:             Optional[str]   = None,
        manipulation_threshold: float           = 6.0,
        abstract_threshold:     float           = 3.0,
        model:                  str             = "es_core_news_lg",
    ):
        self._model = model

        # ── Load lexicons ─────────────────────────────────────
        self._motility = self._load_motility(motility_path) if motility_path else {}
        self._vad      = self._load_vad(vad_path)           if vad_path      else {}

        # ── Load vectors ──────────────────────────────────────
        if word_vectors is not None:
            self._wv = word_vectors
        elif vectors_path:
            self._wv = self._load_vectors(vectors_path)
        else:
            self._wv = None

        # ── Build centroids ───────────────────────────────────

        # f21 — manipulable objects (high concreteness words from norms)
        # Requires norms_path — returns -1 for all words if not provided
        if norms_path:
            manip_words = self._load_manipulation_words(
                norms_path, manipulation_threshold
            )
            self._manip_centroid = self._build_centroid(manip_words)
        else:
            print("  [f21] norms_path not provided — f21 will return -1")
            self._manip_centroid = None

        # f22 — motor actions (weighted centroid from motility norms)
        # Falls back to a small default list only if motility norms are missing
        if self._motility:
            self._motor_centroid = self._build_weighted_centroid(
                self._motility, threshold=4.0
            )
        else:
            print("  [f22] motility_path not provided — f22 will return -1")
            self._motor_centroid = None

        # f23 — abstract cognition (low concreteness words from norms)
        # Requires norms_path — returns -1 for all words if not provided
        if norms_path:
            abstr_words = self._load_abstract_words(
                norms_path, abstract_threshold
            )
            self._abstr_centroid = self._build_centroid(abstr_words)
        else:
            print("  [f23] norms_path not provided — f23 will return -1")
            self._abstr_centroid = None

    # ── Public API ────────────────────────────────────────────

    def extract_text(self, text: str) -> pd.DataFrame:
        nlp     = _get_nlp(self._model)
        doc     = nlp(text)
        records = [self._extract_token(t) for t in doc if t.pos_ in CONTENT_POS]
        return pd.DataFrame(records)

    def extract_file(self, path: str, encoding: str = "utf-8") -> pd.DataFrame:
        text = open(path, encoding=encoding).read().strip()
        return self.extract_text(text)

    def coverage(self, text: str) -> dict:
        nlp     = _get_nlp(self._model)
        doc     = nlp(text)
        content = [t for t in doc if t.pos_ in CONTENT_POS]
        n       = len(content) if content else 1
        return {
            "motility": sum(t.lemma_.lower() in self._motility for t in content) / n,
            "vad":      sum(t.lemma_.lower() in self._vad      for t in content) / n,
            "vectors":  sum(
                self._wv is not None and t.lemma_.lower() in self._wv
                for t in content
            ) / n,
        }

    # ── Token-level ───────────────────────────────────────────

    def _extract_token(self, token: spacy.tokens.Token) -> dict:
        lemma    = token.lemma_.lower()
        motility = self._motility.get(lemma, 0.0)
        valence  = self._vad.get(lemma, {}).get("valence")

        return {
            "form":  token.text,
            "lemma": lemma,
            "pos":   token.pos_,
            "f20_motor_content":      round(motility, 4),
            "f21_dist_manipulation":  self._cosine_to_centroid(lemma, self._manip_centroid),
            "f22_dist_motor_action":  self._cosine_to_centroid(lemma, self._motor_centroid),
            "f23_dist_abstract":      self._cosine_to_centroid(lemma, self._abstr_centroid),
            "f24_emotion_motor":      round(valence * motility, 4)
                                      if valence is not None else NOT_FOUND,
        }

    # ── Norm-based word list loaders ──────────────────────────

    def _load_manipulation_words(self, path: str, threshold: float) -> list[str]:
        """
        Load words with concreteness >= threshold from norms CSV.
        These form the manipulable-objects cluster centroid for f21.
        """
        df    = pd.read_csv(path)
        df["word"] = df["word"].str.lower().str.strip()
        words = df.loc[df["concreteness"] >= threshold, "word"].tolist()

        if not words:
            raise ValueError(
                f"No words found with concreteness >= {threshold} in '{path}'.\n"
                f"Lower the manipulation_threshold or check the norms file.\n"
                f"Concreteness range in file: "
                f"{df['concreteness'].min():.2f} - {df['concreteness'].max():.2f}"
            )

        print(f"  [f21] {len(words)} manipulation words (concreteness ≥ {threshold})")
        return words

    def _load_abstract_words(self, path: str, threshold: float) -> list[str]:
        """
        Load words with concreteness <= threshold from norms CSV.
        These form the abstract-cognition cluster centroid for f23.
        """
        df    = pd.read_csv(path)
        df["word"] = df["word"].str.lower().str.strip()
        words = df.loc[df["concreteness"] <= threshold, "word"].tolist()

        if not words:
            raise ValueError(
                f"No words found with concreteness <= {threshold} in '{path}'.\n"
                f"Raise the abstract_threshold or check the norms file.\n"
                f"Concreteness range in file: "
                f"{df['concreteness'].min():.2f} - {df['concreteness'].max():.2f}"
            )

        print(f"  [f23] {len(words)} abstract words (concreteness ≤ {threshold})")
        return words

    # ── Vector helpers ────────────────────────────────────────

    def _build_centroid(self, words: list[str]) -> Optional[np.ndarray]:
        """Mean embedding of a word list. Warns if coverage is low."""
        if self._wv is None or not words:
            return None

        vecs = [self._wv[w] for w in words if w in self._wv]

        if not vecs:
            print(f"  [warn] No words from the list found in vector vocabulary.")
            return None

        coverage = len(vecs) / len(words)
        if coverage < 0.5:
            print(f"  [warn] Low centroid coverage: {coverage:.1%} of words found in vocab")

        return np.mean(vecs, axis=0)

    def _build_weighted_centroid(
        self, motility_dict: dict, threshold: float
    ) -> Optional[np.ndarray]:
        """
        Weighted mean embedding of verbs above a motility threshold.
        Weight = motility score. Used for f22 motor-action centroid.
        """
        if self._wv is None:
            return None

        vecs, weights = [], []
        for word, score in motility_dict.items():
            if score >= threshold and word in self._wv:
                vecs.append(self._wv[word])
                weights.append(score)

        if not vecs:
            print("  [f22] No motility words found in vocab — f22 will return -1")
            return None

        weights = np.array(weights)
        weights /= weights.sum()
        centroid = np.average(vecs, axis=0, weights=weights)
        print(f"  [f22] Motor centroid built from {len(vecs)} verbs (motility ≥ {threshold})")
        return centroid

    def _cosine_to_centroid(
        self, lemma: str, centroid: Optional[np.ndarray]
    ) -> float:
        if self._wv is None or centroid is None:
            return NOT_FOUND
        if lemma not in self._wv:
            return NOT_FOUND
        from scipy.spatial.distance import cosine
        return round(float(cosine(self._wv[lemma], centroid)), 6)

    # ── Resource loaders ──────────────────────────────────────

    @staticmethod
    def _load_motility(path: str) -> dict:
        df = pd.read_csv(path)
        result = dict(zip(df["word"].str.lower(), df["score"].astype(float)))
        print(f"  Motility lexicon loaded: {len(result):,} entries from '{path}'")
        return result

    @staticmethod
    def _load_vad(path: str) -> dict:
        df = pd.read_csv(path)
        result = {
            row["word"].lower(): {
                "valence":   row["valence"],
                "arousal":   row["arousal"],
                "dominance": row["dominance"],
            }
            for _, row in df.iterrows()
        }
        print(f"  VAD lexicon loaded: {len(result):,} entries from '{path}'")
        return result

    @staticmethod
    def _load_vectors(path: str):
        """
        Load word vectors from any supported format.
        .bin with 'cc.es' in name → fastText
        .bin otherwise             → word2vec binary
        .vec / .txt                → word2vec text
        .model                     → gensim native Word2Vec
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