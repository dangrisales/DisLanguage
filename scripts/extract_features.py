"""
extract_features.py
====================
Extract all DisLanguage features from a corpus of Spanish transcripts.

Runs all five DisLanguage extractors on every .txt file and writes CSVs.

Output
------
    output_dir/
        {subject_id}_features.csv    ← one per subject, word-level (f01-f30)
        all_subjects_features.csv    ← all subjects combined, word-level
        all_subjects_discourse.csv   ← one row per subject, text-level features

Usage
-----
    python extract_features.py \\
        --transcriptions  data/PC-GITA/transcriptions \\
        --output-dir      data/PC-GITA/features \\
        --resources       resources/

    # Skip subjects already processed (resume interrupted run)
    python extract_features.py ... --skip-existing

    # Disable specific groups
    python extract_features.py ... --no-lexical --no-semantic
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract all DisLanguage features from a corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--transcriptions", required=True,
                   help="Directory containing .txt transcription files")
    p.add_argument("--output-dir",     required=True,
                   help="Directory where CSV feature files will be saved")
    p.add_argument("--resources",      default="resources/",
                   help="Directory containing external lexicon files")
    p.add_argument("--vectors",        default=None,
                   help=(
                       "Path to word vector file. Supported formats:\n"
                       "  .bin   → fastText (cc.es.300.bin) or word2vec binary\n"
                       "  .vec   → word2vec text format\n"
                       "  .model → gensim native Word2Vec\n"
                       "If not set, auto-detects from resources/ in this order:\n"
                       "  cc.es.300.bin, word2vec_es.bin, word2vec_es.vec, word2vec_es.model"
                   ))
    p.add_argument("--skip-existing",  action="store_true",
                   help="Skip subjects that already have a feature CSV")

    # Toggle groups on/off
    p.add_argument("--no-morphological", action="store_true")
    p.add_argument("--no-lexical",       action="store_true")
    p.add_argument("--no-semantic",      action="store_true")
    p.add_argument("--no-affective",     action="store_true")
    p.add_argument("--no-discourse",     action="store_true",
                   help="Skip text-level discourse features")
    p.add_argument("--no-surprisal",     action="store_true",
                   help="Skip f19 surprisal (slow — GPT-2 runs word by word)")

    return p.parse_args()


# ── Format-aware vector loader ────────────────────────────────────────────────

def load_vectors(path: str):
    """
    Load word vectors from any supported format.

    Formats
    -------
    .bin    → tries fastText first (supports subwords + OOV),
              falls back to word2vec binary if not a fastText file
    .vec    → word2vec text format
    .txt    → word2vec text format
    .model  → gensim native Word2Vec format
    """
    from gensim.models import KeyedVectors, Word2Vec

    print(f"  Loading word vectors from '{path}' (this may take a while)...")

    if path.endswith(".model"):
        wv = Word2Vec.load(path).wv
        fmt = "gensim Word2Vec"

    elif path.endswith(".bin"):
        # Try fastText first — fastText .bin files support subword OOV,
        # giving better coverage for rare clinical Spanish terms.
        try:
            from gensim.models.fasttext import load_facebook_model
            wv  = load_facebook_model(path).wv
            fmt = "fastText"
        except Exception:
            # Not a fastText binary — load as word2vec binary
            wv  = KeyedVectors.load_word2vec_format(path, binary=True)
            fmt = "word2vec binary"

    elif path.endswith((".vec", ".txt")):
        wv  = KeyedVectors.load_word2vec_format(path, binary=False)
        fmt = "word2vec text"

    else:
        raise ValueError(
            f"Unrecognised vector file extension: '{path}'\n"
            "Supported: .bin (fastText or word2vec), .vec, .txt, .model"
        )

    print(f"  Word vectors loaded: {len(wv):,} entries  [{fmt}]")
    return wv


def find_vectors(resources: Path, vectors_arg: str | None) -> Path | None:
    """
    Resolve the vector file path.

    Priority:
    1. Explicit --vectors argument
    2. Auto-detect from resources/ in a fixed preference order
    """
    if vectors_arg:
        p = Path(vectors_arg)
        if not p.exists():
            print(f"  [warn] --vectors file not found: '{p}' — skipping vectors")
            return None
        return p

    # Auto-detect — fastText preferred, then Word2Vec variants
    candidates = [
        resources / "cc.es.300.bin",        # fastText (default)
        resources / "word2vec_es.bin",       # word2vec binary
        resources / "word2vec_es.vec",       # word2vec text
        resources / "word2vec_es.model",     # gensim native
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


# ── Extractor initialisation ──────────────────────────────────────────────────

def init_extractors(resources: Path, args) -> dict:
    """
    Load all extractors and their resources once.
    Vectors are loaded once and shared between Lexical and Semantic.
    Returns a dict of active extractors.
    """
    extractors = {}

    # ── Load word vectors once (shared by Lexical and Semantic) ──────────
    wv = None
    needs_vectors = not args.no_lexical or not args.no_semantic

    if needs_vectors:
        vectors_path = find_vectors(resources, args.vectors)
        if vectors_path:
            wv = load_vectors(str(vectors_path))
        else:
            print(
                "  [warn] No word vector file found — f11, f12, f21, f22, f23 will be -1\n"
                "         Place cc.es.300.bin or word2vec_es.bin in resources/\n"
                "         or pass --vectors /path/to/vectors.bin"
            )

    # ── Morphological (f01-f10) ───────────────────────────────
    if not args.no_morphological:
        from dislanguage import Morphological

        morfessor_path = resources / "es_morfessor.bin"
        extractors["morphological"] = Morphological(
            morfessor_path=str(morfessor_path) if morfessor_path.exists() else None,
        )
        print("  Morphological extractor ready")

    # ── Lexical (f11-f19) ─────────────────────────────────────
    if not args.no_lexical:
        from dislanguage import Lexical

        freq_path  = resources / "SUBTLEX-ESP.csv"
        norms_path = resources / "norms_combined.csv"

        lex_ext = Lexical(
            freq_path=str(freq_path)     if freq_path.exists()  else None,
            norms_path=str(norms_path)   if norms_path.exists() else None,
            word_vectors=wv,             # shared — no reload
        )

        if not args.no_surprisal:
            lex_ext.load_lm()

        extractors["lexical"] = lex_ext
        print("  Lexical extractor ready")

    # ── Semantic (f20-f24) ────────────────────────────────────
    if not args.no_semantic:
        from dislanguage import Semantic

        motility_path = resources / "motility_scores.csv"
        vad_path      = resources / "NRC-VAD-es.csv"
        norms_path    = resources / "norms_combined.csv"

        extractors["semantic"] = Semantic(
            motility_path=str(motility_path) if motility_path.exists() else None,
            vad_path=str(vad_path)           if vad_path.exists()      else None,
            norms_path=str(norms_path)       if norms_path.exists()    else None,
            word_vectors=wv,
        )
        print("  Semantic extractor ready")

    # ── Affective (f25-f30) ───────────────────────────────────
    if not args.no_affective:
        from dislanguage import Affective

        vad_path = resources / "NRC-VAD-es.csv"
        if not vad_path.exists():
            print(f"  [warn] NRC-VAD-es.csv not found — affective features will be skipped")
        else:
            extractors["affective"] = Affective(
                vad_path=str(vad_path),
            )
            print("  Affective extractor ready")

    # ── Discourse (text-level) ────────────────────────────────
    if not args.no_discourse:
        from dislanguage import Discourse
        extractors["discourse"] = Discourse()
        print("  Discourse extractor ready")

    return extractors


# ── Single subject pipeline ───────────────────────────────────────────────────

def extract_subject(
    txt_path:   Path,
    subject_id: str,
    extractors: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all active extractors on one transcript.

    Returns
    -------
    word_df : pd.DataFrame
        One row per content word (word-level extractors).
    disc_df : pd.DataFrame
        One row with text-level discourse features (or empty DataFrame).
    """
    text = txt_path.read_text(encoding="utf-8").strip()

    # ── Word-level extractors ─────────────────────────────────
    word_extractors = {k: v for k, v in extractors.items() if k != "discourse"}
    frames = []
    for group, ext in word_extractors.items():
        df = ext.extract_text(text)
        df = df.reset_index(drop=True)
        frames.append((group, df))

    if frames:
        meta_cols    = ["form", "lemma", "pos"]
        first_group, first_df = frames[0]
        feature_cols = [c for c in first_df.columns if c not in meta_cols]
        word_df      = first_df[meta_cols + feature_cols].copy()

        for group, df in frames[1:]:
            feat_cols = [c for c in df.columns if c not in meta_cols]
            word_df   = pd.concat([word_df, df[feat_cols]], axis=1)

        word_df.insert(0, "subject_id", subject_id)
    else:
        word_df = pd.DataFrame()

    # ── Discourse extractor (text-level) ──────────────────────
    if "discourse" in extractors:
        disc_df = extractors["discourse"].extract_text(text)
        disc_df.insert(0, "subject_id", subject_id)
    else:
        disc_df = pd.DataFrame()

    return word_df, disc_df


# ── Coverage report ───────────────────────────────────────────────────────────

def print_coverage(df: pd.DataFrame) -> None:
    """Print per-feature coverage across the full corpus."""
    print("\n=== Coverage report ===")
    feature_cols = [
        c for c in df.columns
        if c.startswith("f")
        and pd.api.types.is_numeric_dtype(df[c])
        and c != "f01_pos_label"
    ]
    for col in feature_cols:
        total   = len(df)
        covered = ((df[col] != -1) & (df[col].notna())).sum()
        pct     = covered / total * 100 if total > 0 else 0
        status  = "✓" if pct >= 90 else "~" if pct >= 50 else "✗"
        print(f"  {status} {col}: {covered:,} / {total:,}  ({pct:.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    transcriptions = Path(args.transcriptions)
    output_dir     = Path(args.output_dir)
    resources      = Path(args.resources)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover transcripts
    txt_files = sorted(transcriptions.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {transcriptions}")
        sys.exit(1)
    print(f"\nFound {len(txt_files)} transcripts in {transcriptions}\n")

    # Initialise extractors
    print("Loading extractors and resources...")
    extractors = init_extractors(resources, args)
    if not extractors:
        print("No extractors active — nothing to do.")
        sys.exit(1)
    print(f"\nActive groups: {', '.join(extractors.keys())}\n")

    # Process each subject
    all_records  = []
    disc_records = []
    failed       = []
    fieldnames   = None

    for txt_path in tqdm(txt_files, desc="Extracting features"):
        subject_id = txt_path.stem
        out_csv    = output_dir / f"{subject_id}_features.csv"

        if args.skip_existing and out_csv.exists():
            try:
                all_records.append(pd.read_csv(out_csv))
            except Exception:
                pass
            continue

        try:
            word_df, disc_df = extract_subject(txt_path, subject_id, extractors)

            if word_df.empty and disc_df.empty:
                failed.append((subject_id, "no features extracted"))
                continue

            # ── Word-level CSV
            if not word_df.empty:
                if fieldnames is None:
                    fieldnames = list(word_df.columns)
                word_df.to_csv(out_csv, index=False)
                all_records.append(word_df)

            # ── Discourse row
            if not disc_df.empty:
                disc_records.append(disc_df)

        except Exception as e:
            failed.append((subject_id, str(e)))
            tqdm.write(f"  [error] {subject_id}: {e}")

    # ── Combined word-level CSV
    if all_records:
        combined_path = output_dir / "all_subjects_features.csv"
        combined      = pd.concat(all_records, ignore_index=True)
        combined.to_csv(combined_path, index=False)

        print(f"\nDone.")
        print(f"  Subjects processed : {len(all_records)}")
        print(f"  Total word rows    : {len(combined):,}")
        print(f"  Word-level CSV     : {combined_path}")

        print_coverage(combined)

    # ── Combined discourse CSV
    if disc_records:
        disc_path = output_dir / "all_subjects_discourse.csv"
        disc_all  = pd.concat(disc_records, ignore_index=True)
        # Sort columns alphabetically for consistency
        feat_cols = sorted([c for c in disc_all.columns if c != "subject_id"])
        disc_all  = disc_all[["subject_id"] + feat_cols]
        disc_all.to_csv(disc_path, index=False)
        print(f"  Discourse CSV      : {disc_path}  ({len(disc_all)} subjects × {len(feat_cols)} features)")

    # Failure log
    if failed:
        log_path = output_dir / "failed_subjects.log"
        with open(log_path, "w") as f:
            for subj, reason in failed:
                f.write(f"{subj}\t{reason}\n")
        print(f"\n  {len(failed)} failures logged to {log_path}")
    else:
        print("\n  All subjects processed successfully.")


if __name__ == "__main__":
    main()