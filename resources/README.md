# Resources

This folder contains external lexicons and models required by DisLanguage.
**None of these files are included in the repository** — download each one
following the instructions below.

---

## Quickstart

All resource preparation is handled by a single script:

```bash
# Check what you already have
python scripts/prepare_resources.py --status

# Convert a downloaded file (example)
python scripts/prepare_resources.py --subtlex      resources/raw/SUBTLEX-ESP.xlsx
python scripts/prepare_resources.py --stadthagen2016 resources/raw/13428_2015_684_MOESM1_ESM.xlsx
python scripts/prepare_resources.py --stadthagen2017 resources/raw/stadthagen2017.xlsx
python scripts/prepare_resources.py --motility     resources/raw/motor_content.xlsx
python scripts/prepare_resources.py --vad          resources/raw/Spanish-NRC-VAD-Lexicon.txt

# Generate EsPal query batches → upload → merge
python scripts/prepare_resources.py --espal-query
# ... upload batches to EsPal (see Resource 3 below) ...
python scripts/prepare_resources.py --espal-merge

# Merge all norm sources into norms_combined.csv
python scripts/prepare_resources.py --merge-norms

# Train Morfessor model
python scripts/prepare_resources.py --morfessor
```

---

## Folder structure

```
resources/
    NRC-VAD-es.csv          ← f25–f30, f24
    SUBTLEX-ESP.csv         ← f17
    norms_combined.csv      ← f13, f14, f15
    motility_scores.csv     ← f20, f24
    cc.es.300.bin           ← f11, f12, f21–f23  (fastText, ~8 GB)
    word2vec_es.bin         ← f11, f12, f21–f23  (Word2Vec, optional alternative)
    es_morfessor.bin        ← f10  (trained locally)
    README.md               ← this file
    raw/                    ← original downloaded files (not committed to git)
    │   ├── 13428_2015_684_MOESM1_ESM.xlsx
    │   ├── stadthagen2017_supplementary.xlsx
    │   ├── motor_content.xlsx
    │   ├── SUBTLEX-ESP.xlsx
    │   ├── EsPal_norms.csv
    │   └── written_es_wordlist_out.txt
    └── tmp/                ← intermediate query files
        ├── corpus_vocabulary.txt
        ├── espal_query_batch_1.txt
        ├── espal_query_batch_2.txt
        ├── espal_batch_1_out.txt
        └── espal_batch_2_out.txt
```

---

## Resource 1 — NRC-VAD Spanish lexicon

**Used for:** f25 polarity · f26 valence · f27 arousal · f28 activation · f29 dominance · f30 distance · f24 emotion×motor

**Coverage:** ~20,000 Spanish words · Scale: 0–1

**License:** free for research use (fill in request form)

### Download

1. Go to: https://saifmohammad.com/WebPages/nrc-vad.html
2. Fill in the form and download the zip file
3. Inside the zip find: `OneFilePerLanguage/Spanish-NRC-VAD-Lexicon.txt`
4. Save to `resources/raw/Spanish-NRC-VAD-Lexicon.txt`

### Prepare with script

```bash
python scripts/prepare_resources.py --vad resources/raw/Spanish-NRC-VAD-Lexicon.txt
```

### Manual conversion

```python
import pandas as pd

df = pd.read_csv("resources/raw/Spanish-NRC-VAD-Lexicon.txt", sep="\t")
df = df[["Spanish Word", "Valence", "Arousal", "Dominance"]]
df.columns = ["word", "valence", "arousal", "dominance"]
df = df.dropna(subset=["word"])
df.to_csv("resources/NRC-VAD-es.csv", index=False)
print(f"Saved {len(df):,} entries")
```

---

## Resource 2 — SUBTLEX-ESP

**Used for:** f17 corpus frequency

**Coverage:** ~94,000 Spanish words · Unit: frequency per million words

**License:** free for research use

### Download

1. Go to: https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexesp
2. Download the Excel file
3. Save to `resources/raw/SUBTLEX-ESP.xlsx`

### Prepare with script

```bash
python scripts/prepare_resources.py --subtlex resources/raw/SUBTLEX-ESP.xlsx
```

### Manual conversion

```python
import pandas as pd

df = pd.read_excel("resources/raw/SUBTLEX-ESP.xlsx", sheet_name=0)

# File stores words in 3 column groups — stack them
g1 = df[["Word", "Freq. per million"]].copy()
g2 = df[["Word.1", "Freq. per million.1"]].copy()
g3 = df[["Word.2", "Freq. per million.2"]].copy()
g1.columns = g2.columns = g3.columns = ["word", "freq_pm"]

result = pd.concat([g1, g2, g3], ignore_index=True)
result = result.dropna(subset=["word"])
result["word"] = result["word"].str.lower().str.strip()
result.to_csv("resources/SUBTLEX-ESP.csv", index=False)
print(f"Saved {len(result):,} entries")
```

---

## Resource 3 — Psycholinguistic norms

**Used for:** f13 familiarity · f14 imageability · f15 concreteness

**Coverage:** ~4,300 words (three sources merged)

Three sources are merged in priority order: EsPal → Stadthagen 2017 → Stadthagen 2016.

---

### Source A — EsPal web interface (largest coverage)

EsPal does not offer a bulk download — you query it with a word list and download the results.

**Step 1 — Generate query batches**

```bash
python scripts/prepare_resources.py --espal-query
```

This generates batches in `resources/tmp/` (max 9,500 words each, EsPal limit is 10,000).

**Step 2 — Upload each batch to EsPal**

1. Go to: http://www.bcbl.eu/databases/espal/
2. Click **Words to Properties**
3. Upload `resources/tmp/espal_query_batch_1.txt`
4. Expand **Subjective Ratings** → check Familiarity, Imageability, Concreteness
5. Click **Submit** and download the result
6. Save as `resources/tmp/espal_batch_1_out.txt`
7. Repeat for each batch

**Step 3 — Merge results**

```bash
python scripts/prepare_resources.py --espal-merge
```

---

### Source B — Stadthagen-Gonzalez et al. (2016)

**Coverage:** 1,400 Spanish words · Scale: 1–7

1. Go to: https://link.springer.com/article/10.3758/s13428-015-0684-y
2. Click **Supplementary material** → download `13428_2015_684_MOESM1_ESM.xlsx`
3. Save to `resources/raw/13428_2015_684_MOESM1_ESM.xlsx`

```bash
python scripts/prepare_resources.py --stadthagen2016 resources/raw/13428_2015_684_MOESM1_ESM.xlsx
```

---

### Merge all norm sources

```bash
python scripts/prepare_resources.py --merge-norms
```

This merges EsPal → Stadthagen 2017 → Stadthagen 2016 in priority order
and saves the result to `resources/norms_combined.csv`.

---

## Resource 4 — Motor content norms

**Used for:** f20 motor content · f24 emotion×motor

**Coverage:** 4,565 Spanish verbs · Scale: 1–7

**Source:** San Miguel Abella & González-Nosti (2020)

**License:** free for research use

### Download

1. Go to: https://inco.grupos.uniovi.es/enlaces
2. Download **"Motor content norms for 4,565 verbs in Spanish"**
3. Save to `resources/raw/motor_content.xlsx`

### Prepare with script

```bash
python scripts/prepare_resources.py --motility resources/raw/motor_content.xlsx
```

### Manual conversion

```python
import pandas as pd

df = pd.read_excel("resources/raw/motor_content.xlsx")
result = df[["Verbs", "Average motor content"]].copy()
result.columns = ["word", "score"]
result["word"] = result["word"].str.lower().str.strip()
result = result.dropna(subset=["word", "score"])
result.to_csv("resources/motility_scores.csv", index=False)
print(f"Saved {len(result):,} entries")
```

---

## Resource 5 — Word vectors

**Used for:** f11 local semantic variability · f12 global semantic variability · f21 distance to manipulation · f22 distance to motor action · f23 distance to abstract cognition

Two options — fastText is recommended for clinical Spanish corpora (better OOV coverage via subwords). Pass the desired file with `--vectors` when running `extract_features.py`.

---

### Option A — fastText (recommended)

**Coverage:** 2,000,000 Spanish words + subword OOV · Dimensions: 300

**Size:** ~7 GB compressed, ~8 GB uncompressed

**License:** Creative Commons Attribution-Share-Alike 3.0

```bash
cd resources/
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.bin.gz
gunzip cc.es.300.bin.gz
# Result: resources/cc.es.300.bin (~8 GB)
```

```bash
pip install gensim scipy
```

```bash
python scripts/extract_features.py ... --vectors resources/cc.es.300.bin
```

---

### Option B — Word2Vec SBW (alternative)

**Coverage:** ~1,000,000 Spanish words · Dimensions: 300

**Size:** ~1.2 GB

**Source:** Cardellino (2016) Spanish Billion Words Corpus

```bash
# Download from: http://crscardellino.github.io/SBWCE/
# File: SBW-vectors-300-min5.bin.gz
wget http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.bin.gz
gunzip SBW-vectors-300-min5.bin.gz
mv SBW-vectors-300-min5.bin resources/word2vec_es.bin
```

```bash
python scripts/extract_features.py ... --vectors resources/word2vec_es.bin
```

**Note:** fastText covers ~3.4% more words in the PC-GITA corpus due to subword handling
of regional Spanish terms. See `scripts/compare_vectors.py` to compare both models.

---

## Resource 6 — Morfessor model

**Used for:** f10 morpheme count

**Note:** trained locally from SUBTLEX-ESP — no external download needed.
Run once after preparing SUBTLEX-ESP.

### Train with script

```bash
python scripts/prepare_resources.py --morfessor
```

### Manual training

```python
import pandas as pd
import morfessor

df = pd.read_csv("resources/SUBTLEX-ESP.csv")
df = df.nlargest(30000, "freq_pm")

word_counts = list(zip(df["freq_pm"].astype(int).clip(lower=1), df["word"]))

io    = morfessor.MorfessorIO()
model = morfessor.BaselineModel()
model.load_data(word_counts, count_modifier=lambda x: 1)
model.train_batch()

io.write_binary_model_file("resources/es_morfessor.bin", model)
print("Saved to resources/es_morfessor.bin")
```

### Verify

```python
for word in ["rápidamente", "trabajador", "imposible", "corriendo"]:
    segs = model.viterbi_segment(word)[0]
    print(f"  {word} → {segs}  ({len(segs)} morphemes)")
```

Expected:
```
  rápidamente → ['rá', 'pida', 'mente']  (3 morphemes)
  trabajador  → ['trabaja', 'dor']       (2 morphemes)
  imposible   → ['im', 'posible']        (2 morphemes)
  corriendo   → ['cor', 'riendo']        (2 morphemes)
```

---

## Resource 7 — GPT-2 Spanish

**Used for:** f19 word surprisal

**Size:** ~500 MB · Downloaded automatically from HuggingFace on first use

**License:** MIT

### Install

```bash
pip install transformers torch
```

### Usage

The model (`datificate/gpt2-small-spanish`) downloads and caches automatically
in `~/.cache/huggingface/` on the first call to `ext.load_lm()`.

```python
from dislanguage import Lexical

ext = Lexical(...)
ext.load_lm()   # downloads ~500 MB on first call, cached after
```

To skip surprisal during extraction (saves ~14 minutes per run):

```bash
python scripts/extract_features.py ... --no-surprisal
```

---

## Summary

| File | Features | Source | Size |
|---|---|---|---|
| NRC-VAD-es.csv | f24–f30 | saifmohammad.com | ~2 MB |
| SUBTLEX-ESP.csv | f17 | ugent.be | ~2 MB |
| norms_combined.csv | f13–f15 | EsPal + Stadthagen 2016/2017 | ~1 MB |
| motility_scores.csv | f20, f24 | inco.grupos.uniovi.es | ~200 KB |
| cc.es.300.bin | f11, f12, f21–f23 | fasttext.cc | ~8 GB |
| word2vec_es.bin | f11, f12, f21–f23 | crscardellino.github.io | ~1.2 GB |
| es_morfessor.bin | f10 | trained locally | ~1 MB |
| GPT-2 Spanish | f19 | HuggingFace (auto) | ~500 MB |