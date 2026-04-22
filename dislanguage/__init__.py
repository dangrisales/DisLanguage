"""
DisLanguage
===========
A Python toolkit to extract interpretable word-level language features
for neurodegenerative disorder research (e.g., Parkinson's disease).

Feature groups
--------------
Morphological  (f01-f10)  POS, consonants, vowels, chars, phonemes,
                           syllables, tense, person, number, morphemes
Lexical        (f11-f19)  Semantic variability, familiarity, imageability,
                           concreteness, polysemy, frequency, surprisal
Semantic       (f20-f24)  Motor content, cluster distances, emotionxmotor
Affective      (f25-f30)  Valence, arousal, dominance, polarity, activation

Quick start
-----------
    from dislanguage import Morphological, Lexical, Semantic, Affective

    ext = Morphological(morfessor_path="resources/es_morfessor.bin")
    df  = ext.extract_text("El médico corre rápidamente.")
    df.to_csv("features_morphological.csv", index=False)
"""

from .morphological import Morphological
from .affective     import Affective
from .lexical       import Lexical
from .semantic      import Semantic
from .discourse     import Discourse

__all__    = ["Morphological", "Affective", "Lexical", "Semantic", "Discourse"]
__version__ = "0.1.0"