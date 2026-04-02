"""
prepare_resources.py
=====================
Utility scripts to convert and prepare all external resources needed
by DisLanguage. Run each section independently as needed.

Usage
-----
    # Prepare NRC-VAD lexicon
    python scripts/prepare_resources.py --vad path/to/Spanish-NRC-VAD-Lexicon.txt

    # Prepare SUBTLEX-ESP
    python scripts/prepare_resources.py --subtlex path/to/SUBTLEX-ESP.xlsx

    # Prepare Stadthagen 2016 norms
    python scripts/prepare_resources.py --stadthagen2016 path/to/13428_2015_684_MOESM1_ESM.xlsx

    # Prepare Stadthagen 2017 norms
    python scripts/prepare_resources.py --stadthagen2017 path/to/stadthagen2017.xlsx

    # Query EsPal — generate word list batches for upload
    python scripts/prepare_resources.py --espal-query

    # Merge EsPal results after uploading batches
    python scripts/prepare_resources.py --espal-merge

    # Prepare motor content norms (motility)
    python scripts/prepare_resources.py --motility path/to/Appendix1.xlsx

    # Merge all norms into norms_combined.csv
    python scripts/prepare_resources.py --merge-norms

    # Train Morfessor model
    python scripts/prepare_resources.py --morfessor

    # Run all steps that can be automated
    python scripts/prepare_resources.py --all
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


# ── Paths ─────────────────────────────────────────────────────────────────────

RESOURCES = Path("resources")
RAW       = RESOURCES / "raw"
TMP       = RESOURCES / "tmp"


def ensure_dirs():
    RESOURCES.mkdir(exist_ok=True)
    RAW.mkdir(exist_ok=True)
    TMP.mkdir(exist_ok=True)


# ── NRC-VAD ───────────────────────────────────────────────────────────────────

def prepare_vad(input_path: str) -> None:
    """
    Convert NRC-VAD Spanish lexicon to CSV.

    Download from: https://saifmohammad.com/WebPages/nrc-vad.html
    File inside zip: OneFilePerLanguage/Spanish-NRC-VAD-Lexicon.txt
    """
    print(f"\nPreparing NRC-VAD from '{input_path}'...")

    df = pd.read_csv(input_path, sep="\t")
    print(f"  Columns found: {df.columns.tolist()}")

    # Handle different column name variants
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "spanish" in cl or "word" in cl:
            col_map[col] = "word"
        elif "valence" in cl:
            col_map[col] = "valence"
        elif "arousal" in cl:
            col_map[col] = "arousal"
        elif "dominance" in cl:
            col_map[col] = "dominance"

    df = df.rename(columns=col_map)
    df = df[["word", "valence", "arousal", "dominance"]].copy()
    df["word"] = df["word"].str.lower().str.strip()
    df = df.dropna(subset=["word"])

    out = RESOURCES / "NRC-VAD-es.csv"
    df.to_csv(out, index=False)
    print(f"  Saved {len(df):,} entries → {out}")


# ── SUBTLEX-ESP ───────────────────────────────────────────────────────────────

def prepare_subtlex(input_path: str) -> None:
    """
    Convert SUBTLEX-ESP Excel to CSV.

    Download from:
    https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexesp
    """
    print(f"\nPreparing SUBTLEX-ESP from '{input_path}'...")

    df = pd.read_excel(input_path, sheet_name=0)
    print(f"  Columns found: {df.columns.tolist()[:6]}...")

    # File stores words in 3 column groups — stack them
    g1 = df[["Word", "Freq. per million"]].copy()
    g2 = df[["Word.1", "Freq. per million.1"]].copy()
    g3 = df[["Word.2", "Freq. per million.2"]].copy()
    g1.columns = g2.columns = g3.columns = ["word", "freq_pm"]

    result = pd.concat([g1, g2, g3], ignore_index=True)
    result = result.dropna(subset=["word"])
    result["word"] = result["word"].str.lower().str.strip()
    result = result[result["word"].str.isalpha()]

    out = RESOURCES / "SUBTLEX-ESP.csv"
    result.to_csv(out, index=False)
    print(f"  Saved {len(result):,} entries → {out}")


# ── STADTHAGEN 2016 ───────────────────────────────────────────────────────────

def prepare_stadthagen2016(input_path: str) -> None:
    """
    Convert Stadthagen-Gonzalez et al. (2016) supplementary file to CSV.

    Download from:
    https://link.springer.com/article/10.3758/s13428-015-0684-y
    → Supplementary material → 13428_2015_684_MOESM1_ESM.xlsx
    """
    print(f"\nPreparing Stadthagen 2016 from '{input_path}'...")

    df = pd.read_excel(input_path)
    print(f"  Columns found: {df.columns.tolist()}")

    result = df[["Word", "FAM_M", "IMA_M", "CON_M"]].copy()
    result.columns = ["word", "familiarity", "imageability", "concreteness"]
    result["word"] = result["word"].str.lower().str.strip()
    result = result.dropna(subset=["word"])

    out = RAW / "stadthagen2016_norms.csv"
    result.to_csv(out, index=False)
    print(f"  Saved {len(result):,} entries → {out}")


# ── STADTHAGEN 2017 ───────────────────────────────────────────────────────────

def prepare_stadthagen2017(input_path: str) -> None:
    """
    Convert Stadthagen-Gonzalez et al. (2017) supplementary file to CSV.

    Download from:
    https://link.springer.com/article/10.3758/s13428-016-0734-0
    → Supplementary material
    """
    print(f"\nPreparing Stadthagen 2017 from '{input_path}'...")

    df = pd.read_excel(input_path)
    print(f"  Columns found: {df.columns.tolist()}")
    print(df.head(3).to_string())

    # Auto-detect columns
    col_map   = {}
    cols_lower = {c.lower(): c for c in df.columns}

    for key, variants in {
        "word":          ["word", "palabra", "words"],
        "familiarity":   ["fam_m", "familiarity", "familiaridad", "fam"],
        "imageability":  ["ima_m", "imageability", "imaginabilidad", "img_m", "ima"],
        "concreteness":  ["con_m", "concreteness", "concreción", "concreteness_m"],
    }.items():
        for v in variants:
            if v in cols_lower:
                col_map[cols_lower[v]] = key
                break

    missing = [k for k in ["word", "familiarity", "imageability", "concreteness"]
               if k not in col_map.values()]
    if missing:
        print(f"\n  [warn] Could not auto-detect columns for: {missing}")
        print(f"  Please check the column names above and update col_map manually.")
        return

    result = df.rename(columns=col_map)[
        ["word", "familiarity", "imageability", "concreteness"]
    ].copy()
    result["word"] = result["word"].str.lower().str.strip()
    result = result.dropna(subset=["word"])

    out = RAW / "stadthagen2017_norms.csv"
    result.to_csv(out, index=False)
    print(f"  Saved {len(result):,} entries → {out}")


# ── ESPAL QUERY ───────────────────────────────────────────────────────────────

def prepare_espal_query() -> None:
    """
    Generate word list batches to upload to EsPal for familiarity,
    imageability, and concreteness norms.

    Upload batches to: http://www.bcbl.eu/databases/espal/
    Click: Words to Properties → Subjective Ratings
           → Familiarity, Imageability, Concreteness → Submit
    Save results as: resources/tmp/espal_batch_1_out.txt, etc.
    """
    print("\nGenerating EsPal query batches...")

    subtlex_path = RESOURCES / "SUBTLEX-ESP.csv"
    if not subtlex_path.exists():
        print("  [error] SUBTLEX-ESP.csv not found — run --subtlex first")
        return

    subtlex = pd.read_csv(subtlex_path)

    # Load existing coverage to avoid re-querying
    existing_words = set()
    norms_path = RESOURCES / "norms_combined.csv"
    if norms_path.exists():
        existing = pd.read_csv(norms_path)
        existing_words = set(existing["word"].str.lower().str.strip())
        print(f"  Already covered: {len(existing_words):,} words")

    # Load corpus vocabulary if available
    corpus_words = set()
    vocab_path = RESOURCES / "tmp" / "corpus_vocabulary.txt"
    if not vocab_path.exists():
        vocab_path = RESOURCES / "corpus_vocabulary.txt"
    if vocab_path.exists():
        corpus_words = {
            line.strip().lower() for line in vocab_path.read_text().splitlines()
            if line.strip()
        }
        print(f"  Corpus vocabulary: {len(corpus_words):,} words")

    # Top SUBTLEX words not yet covered
    candidates = (
        subtlex
        .nlargest(15000, "freq_pm")["word"]
        .str.lower().str.strip()
        .tolist()
    )
    extra = [
        w for w in candidates
        if w.isalpha() and len(w) > 2 and w not in existing_words
    ]

    # Corpus words not yet covered
    corpus_missing = [
        w for w in corpus_words
        if w.isalpha() and len(w) > 2 and w not in existing_words
    ]

    # Corpus words first (higher priority)
    all_words = list(dict.fromkeys(corpus_missing + extra))

    print(f"\n  Words to query:")
    print(f"    From corpus (not yet covered) : {len(corpus_missing):,}")
    print(f"    From SUBTLEX top-15k          : {len(extra):,}")
    print(f"    Total unique                  : {len(all_words):,}")

    # EsPal limit: 10,000 rows per upload
    BATCH_SIZE = 9500
    batches = [
        all_words[i:i + BATCH_SIZE]
        for i in range(0, len(all_words), BATCH_SIZE)
    ]

    TMP.mkdir(exist_ok=True)
    for i, batch in enumerate(batches, 1):
        out = TMP / f"espal_query_batch_{i}.txt"
        out.write_text("\n".join(batch), encoding="utf-8")
        print(f"  Batch {i}: {len(batch):,} words → {out}")

    print(f"\n  Total batches: {len(batches)}")
    print("  Upload each to: http://www.bcbl.eu/databases/espal/")
    print("  Words to Properties → Subjective Ratings")
    print("  → Familiarity, Imageability, Concreteness → Submit")
    print(f"  Save results as: resources/tmp/espal_batch_N_out.txt")


# ── ESPAL MERGE ───────────────────────────────────────────────────────────────

def prepare_espal_merge() -> None:
    """
    Merge EsPal result files downloaded after uploading query batches.
    Results are saved to resources/raw/espal_norms.csv.
    """
    print("\nMerging EsPal result files...")

    result_files = sorted(TMP.glob("espal_batch_*_out.txt"))
    if not result_files:
        print(f"  [error] No result files found in {TMP}/")
        print(f"  Expected files matching: espal_batch_*_out.txt")
        print(f"  Upload query batches to EsPal first (run --espal-query)")
        return

    frames = []
    for f in result_files:
        df = pd.read_csv(
            f, sep="\t",
            usecols=["word", "familiarity", "imageability", "concreteness"],
        )
        df["word"] = df["word"].str.lower().str.strip()
        df = df.dropna(subset=["familiarity", "imageability", "concreteness"])
        frames.append(df)
        print(f"  {f.name}: {len(df):,} entries with complete ratings")

    espal = pd.concat(frames, ignore_index=True)
    espal = espal.drop_duplicates(subset="word", keep="first")

    out = RAW / "espal_norms.csv"
    espal.to_csv(out, index=False)
    print(f"\n  Merged EsPal norms: {len(espal):,} entries → {out}")


# ── MOTILITY ──────────────────────────────────────────────────────────────────

def prepare_motility(input_path: str) -> None:
    """
    Convert motor content norms to CSV.

    Download from: https://inco.grupos.uniovi.es/enlaces
    San Miguel Abella & González-Nosti (2020)
    """
    print(f"\nPreparing motility norms from '{input_path}'...")

    df = pd.read_excel(input_path)
    print(f"  Columns found: {df.columns.tolist()}")

    result = df[["Verbs", "Average motor content"]].copy()
    result.columns = ["word", "score"]
    result["word"] = result["word"].str.lower().str.strip()
    result = result.dropna(subset=["word", "score"])

    out = RESOURCES / "motility_scores.csv"
    result.to_csv(out, index=False)
    print(f"  Saved {len(result):,} entries → {out}")


# ── MERGE ALL NORMS ───────────────────────────────────────────────────────────

def merge_norms() -> None:
    """
    Merge all available norm sources into norms_combined.csv.

    Priority order (first wins for each word):
        1. EsPal (largest, most recent query)
        2. Stadthagen 2017
        3. Stadthagen 2016
    """
    print("\nMerging all norm sources...")

    sources = []

    # ── EsPal ─────────────────────────────────────────────────
    espal_path = RAW / "espal_norms.csv"
    if espal_path.exists():
        df = pd.read_csv(espal_path)
        sources.append(("EsPal", df))
        print(f"  EsPal           : {len(df):,} entries")
    else:
        print(f"  EsPal           : not found — run --espal-merge first")

    # ── Stadthagen 2017 ───────────────────────────────────────
    stad2017_path = RAW / "stadthagen2017_norms.csv"
    if stad2017_path.exists():
        df = pd.read_csv(stad2017_path)
        sources.append(("Stadthagen2017", df))
        print(f"  Stadthagen 2017 : {len(df):,} entries")
    else:
        print(f"  Stadthagen 2017 : not found — run --stadthagen2017 first")

    # ── Stadthagen 2016 ───────────────────────────────────────
    stad2016_path = RAW / "stadthagen2016_norms.csv"
    if stad2016_path.exists():
        df = pd.read_csv(stad2016_path)
        sources.append(("Stadthagen2016", df))
        print(f"  Stadthagen 2016 : {len(df):,} entries")
    else:
        print(f"  Stadthagen 2016 : not found — run --stadthagen2016 first")

    if not sources:
        print("\n  [error] No norm sources found. Run the individual prepare steps first.")
        return

    # ── Merge — first source wins per word ────────────────────
    combined = pd.concat([df for _, df in sources], ignore_index=True)
    combined = combined.drop_duplicates(subset="word", keep="first")
    combined = combined.dropna(subset=["word"])
    combined = combined.sort_values("word").reset_index(drop=True)

    out = RESOURCES / "norms_combined.csv"

    # Report before saving
    previous_count = 0
    if out.exists():
        previous_count = len(pd.read_csv(out))

    combined.to_csv(out, index=False)

    print(f"\n  Combined: {len(combined):,} entries → {out}")
    if previous_count:
        print(f"  Change  : {previous_count:,} → {len(combined):,} "
              f"(+{len(combined) - previous_count:,} words)")

    # Per-source contribution
    print(f"\n  Per-source contribution:")
    seen = set()
    for name, df in sources:
        new_words = set(df["word"]) - seen
        seen.update(df["word"])
        print(f"    {name:<20} {len(df):>5} total → {len(new_words):>5} unique new words")

    # Column coverage
    print(f"\n  Coverage per column:")
    for col in ["familiarity", "imageability", "concreteness"]:
        if col in combined.columns:
            n   = combined[col].notna().sum()
            pct = n / len(combined) * 100
            print(f"    {col:<16} {n:,} / {len(combined):,} ({pct:.1f}%)")


# ── MORFESSOR ─────────────────────────────────────────────────────────────────

def train_morfessor() -> None:
    """
    Train a Morfessor morpheme segmentation model on SUBTLEX-ESP vocabulary.
    Saves model to resources/es_morfessor.bin.
    """
    print("\nTraining Morfessor model...")

    subtlex_path = RESOURCES / "SUBTLEX-ESP.csv"
    if not subtlex_path.exists():
        print("  [error] SUBTLEX-ESP.csv not found — run --subtlex first")
        return

    try:
        import morfessor
    except ImportError:
        print("  [error] morfessor not installed — run: pip install morfessor")
        return

    df = pd.read_csv(subtlex_path)
    df = df.nlargest(30000, "freq_pm")
    word_counts = list(zip(df["freq_pm"].astype(int).clip(lower=1), df["word"]))

    io    = morfessor.MorfessorIO()
    model = morfessor.BaselineModel()
    model.load_data(word_counts, count_modifier=lambda x: 1)

    print(f"  Training on {len(word_counts):,} words (top 30k by frequency)...")
    model.train_batch()

    out = RESOURCES / "es_morfessor.bin"
    io.write_binary_model_file(str(out), model)
    print(f"  Model saved → {out}")

    # Verify segmentation
    print("\n  Verification:")
    for word in ["rápidamente", "trabajador", "imposible", "corriendo",
                 "incomprensible", "desayunando"]:
        segs = model.viterbi_segment(word)[0]
        print(f"    {word:<20} → {segs}  ({len(segs)} morphemes)")


# ── SUMMARY ───────────────────────────────────────────────────────────────────

def print_summary() -> None:
    """Print current status of all resources."""
    print("\n=== Resource status ===\n")

    checks = [
        ("NRC-VAD-es.csv",       "f25-f30, f24",  "saifmohammad.com/WebPages/nrc-vad.html"),
        ("SUBTLEX-ESP.csv",      "f17",           "ugent.be/pp/.../subtlexesp"),
        ("norms_combined.csv",   "f13-f15",       "EsPal + Stadthagen"),
        ("motility_scores.csv",  "f20, f24",      "inco.grupos.uniovi.es/enlaces"),
        ("cc.es.300.bin",        "f11,f12,f21-23","fasttext.cc/docs/en/crawl-vectors.html"),
        ("es_morfessor.bin",     "f10",           "train locally — run --morfessor"),
    ]

    for filename, features, source in checks:
        path = RESOURCES / filename
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            # Count entries for CSV files
            if filename.endswith(".csv"):
                try:
                    n = len(pd.read_csv(path))
                    print(f"  ✓ {filename:<25} {features:<15} {n:>6} entries")
                except Exception:
                    print(f"  ✓ {filename:<25} {features:<15} {size_mb:.0f} MB")
            else:
                print(f"  ✓ {filename:<25} {features:<15} {size_mb:.0f} MB")
        else:
            print(f"  ✗ {filename:<25} {features:<15} → {source}")

    # Word2Vec (optional)
    w2v_path = RESOURCES / "word2vec_es.bin"
    if w2v_path.exists():
        size_mb = w2v_path.stat().st_size / 1024 / 1024
        print(f"  ✓ {'word2vec_es.bin':<25} {'f11,f12,f21-23':<15} {size_mb:.0f} MB")
    else:
        print(f"  ~ {'word2vec_es.bin':<25} {'f11,f12,f21-23':<15} optional alternative to fastText")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Prepare all external resources for DisLanguage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--vad",            metavar="FILE",
                   help="Convert NRC-VAD Spanish lexicon")
    p.add_argument("--subtlex",        metavar="FILE",
                   help="Convert SUBTLEX-ESP Excel file")
    p.add_argument("--stadthagen2016", metavar="FILE",
                   help="Convert Stadthagen 2016 supplementary Excel")
    p.add_argument("--stadthagen2017", metavar="FILE",
                   help="Convert Stadthagen 2017 supplementary Excel")
    p.add_argument("--motility",       metavar="FILE",
                   help="Convert motor content norms Excel file")
    p.add_argument("--espal-query",    action="store_true",
                   help="Generate EsPal query batches")
    p.add_argument("--espal-merge",    action="store_true",
                   help="Merge downloaded EsPal result files")
    p.add_argument("--merge-norms",    action="store_true",
                   help="Merge all norm sources into norms_combined.csv")
    p.add_argument("--morfessor",      action="store_true",
                   help="Train Morfessor morpheme model")
    p.add_argument("--status",         action="store_true",
                   help="Print current resource status")
    p.add_argument("--all",            action="store_true",
                   help="Run all automated steps (requires files in resources/raw/)")

    args = p.parse_args()

    if len(sys.argv) == 1:
        print_summary()
        p.print_help()
        return

    ensure_dirs()

    if args.status:
        print_summary()
        return

    if args.all:
        print_summary()

    if args.vad:
        prepare_vad(args.vad)
    if args.subtlex:
        prepare_subtlex(args.subtlex)
    if args.stadthagen2016:
        prepare_stadthagen2016(args.stadthagen2016)
    if args.stadthagen2017:
        prepare_stadthagen2017(args.stadthagen2017)
    if args.motility:
        prepare_motility(args.motility)
    if args.espal_query or args.all:
        prepare_espal_query()
    if args.espal_merge or args.all:
        prepare_espal_merge()
    if args.merge_norms or args.all:
        merge_norms()
    if args.morfessor or args.all:
        train_morfessor()

    print("\nDone.")


if __name__ == "__main__":
    main()