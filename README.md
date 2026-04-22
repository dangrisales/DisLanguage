# DisLanguage

![Image](https://github.com/dangrisales/DisLanguage/blob/main/docs/logo.png?raw=true)

A Python toolkit to extract interpretable **word-level and text-level language features** for
neurodegenerative disorder research (e.g., Parkinson's disease, Alzheimer's disease).

Inspired by [DisVoice](https://github.com/jcvasquezc/DisVoice) — same philosophy, applied to language.

---

## Feature groups

| Group | Features | Level | Description |
|---|---|---|---|
| Morphological | f01–f10 | word | POS, syllables, phonemes, tense, person, number, morphemes |
| Lexical | f11–f19 | word | Semantic variability, norms, frequency, polysemy, surprisal |
| Motor semantic | f20–f24 | word | Motor content, cluster distances, emotion × motor |
| Emotion dynamic | f25–f30 | word | Valence, arousal, dominance and derived scores |
| Discourse | — | text | POS ratios, lexical diversity, sentence length |

Full feature descriptions are in the [Features](#features) section below.

---

## Installation

### Step 1 — System dependency

```bash
# Ubuntu / Debian
sudo apt install espeak-ng

# macOS
brew install espeak-ng
```

### Step 2 — Clone and install

```bash
git clone https://github.com/dangrisales/dislanguage.git
cd DisLanguage
pip install -e .
```

### Step 3 — Python packages

```bash
pip install spacy pyphen pandas phonemizer morfessor \
            nltk gensim scipy transformers torch tqdm
```

### Step 4 — spaCy Spanish model

```bash
python -m spacy download es_core_news_lg
```

### Step 5 — NLTK WordNet data

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Step 6 — External resources

Several features require external lexicons and models that must be downloaded
separately. All files go in the `resources/` folder.

> See **[resources/README.md](resources/README.md)** for detailed download
> and conversion instructions for every resource.
> Use `python scripts/prepare_resources.py --status` to check what you have.

| Resource | Features | Size |
|---|---|---|
| NRC-VAD Spanish | f24–f30 | ~2 MB |
| SUBTLEX-ESP | f17 | ~2 MB |
| Psycholinguistic norms (EsPal + Stadthagen) | f13–f15 | ~1 MB |
| Motor content norms | f20, f24 | ~200 KB |
| fastText cc.es.300.bin *(recommended)* | f11, f12, f21–f23 | ~8 GB |
| Word2Vec SBW *(alternative)* | f11, f12, f21–f23 | ~1.2 GB |
| GPT-2 Spanish | f19 | ~500 MB |
| Morfessor model | f10 | ~1 MB (trained locally) |

---

## Corpus extraction

The recommended way to extract features from a corpus is via `extract_features.py`:

```bash
python scripts/extract_features.py \
    --transcriptions data/PC-GITA/transcriptions \
    --output-dir     data/PC-GITA/features \
    --resources      resources/ \
    --vectors        resources/cc.es.300.bin
```

This produces:

```
data/PC-GITA/features/
    all_subjects_features.csv    ← word-level features (one row per content word)
    all_subjects_discourse.csv   ← text-level features (one row per subject)
    {subject_id}_features.csv    ← per-subject word-level features
```

Useful flags:

```bash
--vectors resources/word2vec_es.bin   # use Word2Vec instead of fastText
--no-surprisal                        # skip f19 (saves ~14 min)
--skip-existing                       # resume an interrupted run
--no-lexical --no-semantic            # disable specific feature groups
```

To compare fastText vs Word2Vec outputs:

```bash
python scripts/compare_vectors.py \
    --a       data/PC-GITA/features_fasttext/all_subjects_features.csv \
    --b       data/PC-GITA/features_w2v/all_subjects_features.csv \
    --label-a fastText \
    --label-b Word2Vec
```

---

## Classification preparation

After extraction, prepare features for classification:

```bash
python scripts/prepare_features.py \
    --input  data/PC-GITA/features/all_subjects_features.csv \
    --output data/PC-GITA/prepared/
```

This produces `X.npy`, `y.npy`, `subject_ids.npy`, `feature_names.npy` and
`static_features.csv` — one row per subject, ready for nested cross-validation.

**Safe to do before the train/test split** (no data leakage):
- Replace -1 sentinels with NaN
- One-hot encode f01 and f07 with fixed category order
- Aggregate word-level features to subject-level (mean + std)
- Extract labels from subject IDs (HC→0, PD→1)

**Goes inside the pipeline** (fitted on training fold only):
- `SimpleImputer` — fills NaN with training fold mean
- `StandardScaler` — normalises to zero mean and unit variance

---

## Quick start (Python API)

```python
from pathlib import Path
from dislanguage import Morphological, Lexical, Semantic, Affective, Discourse

BASE = Path(__file__).parent

morph_ext = Morphological(
    morfessor_path=str(BASE / "resources/es_morfessor.bin"),
)
lex_ext = Lexical(
    freq_path=str(BASE / "resources/SUBTLEX-ESP.csv"),
    norms_path=str(BASE / "resources/norms_combined.csv"),
    vectors_path=str(BASE / "resources/cc.es.300.bin"),
)
lex_ext.load_lm()  # GPT-2 for f19 (~500MB, cached after first run)

sem_ext = Semantic(
    motility_path=str(BASE / "resources/motility_scores.csv"),
    vad_path=str(BASE / "resources/NRC-VAD-es.csv"),
    norms_path=str(BASE / "resources/norms_combined.csv"),
    vectors_path=str(BASE / "resources/cc.es.300.bin"),
)
aff_ext  = Affective(vad_path=str(BASE / "resources/NRC-VAD-es.csv"))
disc_ext = Discourse()

text = "El médico corre rápidamente hacia el paciente."

print(morph_ext.extract_text(text))   # word-level
print(lex_ext.extract_text(text))     # word-level
print(sem_ext.extract_text(text))     # word-level
print(aff_ext.extract_text(text))     # word-level
print(disc_ext.extract_text(text))    # one row — text-level
```

---

## Features

### Morphological & phonological (f01–f10)

| # | Feature | Description | Backend | Values |
|---|---|---|---|---|
| f01 | Content-word type | POS category | spaCy | 0=VERB, 1=NOUN, 2=ADJ, 3=ADV |
| f02 | N consonants | Consonant count | string | int |
| f03 | N vowels | Vowel count | string | int |
| f04 | N characters | Character count | string | int |
| f05 | N phonemes | Phoneme count | phonemizer + espeak-ng | int |
| f06 | N syllables | Syllable count | pyphen (es) | int |
| f07 | Verbal tense | Tense of verb forms | spaCy morph | 0=Pres, 1=Past, 2=Imp, 3=Fut, -1=N/A |
| f08 | Person | Grammatical person | spaCy morph | 1, 2, 3, -1=N/A |
| f09 | Grammatical number | Singular or plural | spaCy morph | 0=Sing, 1=Plur, -1=N/A |
| f10 | N morphemes | Morpheme count | Morfessor | int |

### Lexical-semantic & distributional (f11–f19)

| # | Feature | Description | Backend | Values |
|---|---|---|---|---|
| f11 | Local semantic variability | Std cosine dist. to ±3 context words | fastText / Word2Vec | float, -1=N/A |
| f12 | Global semantic variability | Std cosine dist. to all words in text | fastText / Word2Vec | float, -1=N/A |
| f13 | Familiarity | Psycholinguistic norm | EsPal + Stadthagen | float, -1=N/A |
| f14 | Imageability | Psycholinguistic norm | EsPal + Stadthagen | float, -1=N/A |
| f15 | Concreteness | Psycholinguistic norm | EsPal + Stadthagen | float, -1=N/A |
| f16 | Polysemy | Number of WordNet senses | NLTK omw-1.4 | int, -1=N/A |
| f17 | Corpus frequency | Frequency per million | SUBTLEX-ESP | float, -1=N/A |
| f18 | Within-text frequency | Count / total content words | computed | float |
| f19 | Word surprisal | -log2 P(word \| context) | GPT-2 Spanish | float, -1=N/A |

### Motor semantic (f20–f24)

| # | Feature | Description | Backend | Values |
|---|---|---|---|---|
| f20 | Motor content | Motility score | San Miguel & González-Nosti (2020) | float, 0=not found |
| f21 | Distance to manipulation | Cosine dist. to manipulable-objects cluster | fastText / Word2Vec | float, -1=N/A |
| f22 | Distance to motor action | Cosine dist. to motor-action cluster | fastText / Word2Vec | float, -1=N/A |
| f23 | Distance to abstract cognition | Cosine dist. to abstract-cognition cluster | fastText / Word2Vec | float, -1=N/A |
| f24 | Emotion × motor | valence × motility | NRC-VAD + motility | float, -1=N/A |

### Emotion dynamic (f25–f30)

| # | Feature | Description | Formula | Values |
|---|---|---|---|---|
| f25 | Emotion polarity | Sign of valence | sign(v) | -1=neg, 0=neutral, 1=pos |
| f26 | Valence magnitude | Absolute valence | \|v\| | float, -1=N/A |
| f27 | Arousal magnitude | Absolute arousal | \|a\| | float, -1=N/A |
| f28 | Emotional activation | Valence × arousal | \|v\| × \|a\| | float, -1=N/A |
| f29 | Dominance | Raw dominance score | d | float, -1=N/A |
| f30 | Distance to neutral | Euclidean from origin | √(v²+a²) | float, -1=N/A |

Words not found in lexicons return **-1**.

### Discourse features (text-level)

Extracted by the `Discourse` extractor — returns **one row per transcript**.

| Feature | Description |
|---|---|
| r_verb_content | VERB / N content words |
| r_noun_content | NOUN / N content words |
| r_adj_content | ADJ / N content words |
| r_adv_content | ADV / N content words |
| r_verb_total | VERB / N total tokens |
| r_noun_total | NOUN / N total tokens |
| r_adj_total | ADJ / N total tokens |
| r_adv_total | ADV / N total tokens |
| n_total_tokens | Total token count |
| n_content_words | Total content word count |
| n_sentences | Number of sentences |
| content_ratio | N content / N total |
| type_token_ratio | Unique lemmas / N content words |
| mean_sent_length | Mean tokens per sentence |
| std_sent_length | Std of tokens per sentence |

---

## Scripts

| Script | Description |
|---|---|
| `scripts/extract_features.py` | Extract all features from a corpus directory |
| `scripts/prepare_features.py` | Aggregate word-level features to subject-level for classification |
| `scripts/prepare_resources.py` | Download, convert and merge all external resources |
| `scripts/compare_vectors.py` | Compare fastText vs Word2Vec feature outputs |

---

## Python requirements

| Package | Version | Install |
|---|---|---|
| spacy | ≥ 3.7 | `pip install spacy` |
| pyphen | ≥ 0.14 | `pip install pyphen` |
| pandas | ≥ 2.0 | `pip install pandas` |
| phonemizer | ≥ 3.2 | `pip install phonemizer` |
| morfessor | ≥ 2.0 | `pip install morfessor` |
| nltk | ≥ 3.8 | `pip install nltk` |
| gensim | ≥ 4.0 | `pip install gensim` |
| scipy | ≥ 1.10 | `pip install scipy` |
| transformers | ≥ 4.30 | `pip install transformers` |
| torch | ≥ 2.0 | `pip install torch` |
| tqdm | ≥ 4.0 | `pip install tqdm` |

---

## License

MIT