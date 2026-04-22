"""
prepare_features.py
====================
Prepares DisLanguage features for classification.

Performs only statistics-free, deterministic operations that are safe
to do before the train/test split:
    1. Load word-level features
    2. Replace -1 sentinels with NaN
    3. One-hot encode categorical features (f01, f07)
    4. Aggregate to one static feature vector per subject (mean + std)
    5. Extract labels from subject IDs
    6. Save X, y, subject_ids, feature_names

Output
------
    X.npy             ← feature matrix  (n_subjects, n_features)
    y.npy             ← labels          (n_subjects,)  0=HC, 1=PD
    subject_ids.npy   ← subject ID strings
    feature_names.npy ← feature column names (for interpretability)

Usage
-----
    python prepare_features.py \\
        --input  data/PC-GITA/features/all_subjects_features.csv \\
        --output data/PC-GITA/prepared/

    # Then in your training script:
    X            = np.load("data/PC-GITA/prepared/X.npy")
    y            = np.load("data/PC-GITA/prepared/y.npy")
    subject_ids  = np.load("data/PC-GITA/prepared/subject_ids.npy", allow_pickle=True)
    feature_names= np.load("data/PC-GITA/prepared/feature_names.npy", allow_pickle=True)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Label extraction ──────────────────────────────────────────────────────────

def extract_label(subject_id: str) -> int:
    """
    Extract binary label from PC-GITA subject ID.
    HC_* → 0 (healthy control)
    PD_* → 1 (Parkinson's disease)
    """
    sid = subject_id.upper()
    if sid.startswith("HC"):
        return 0
    if sid.startswith("PD"):
        return 1
    raise ValueError(
        f"Cannot extract label from subject ID '{subject_id}'. "
        "Expected prefix 'HC' or 'PD'."
    )


# ── Main preparation ──────────────────────────────────────────────────────────

def prepare(input_csv: str, output_dir: str) -> None:

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"  {len(df):,} word rows · {df['subject_id'].nunique()} subjects")

    # ── 2. Replace -1 sentinel with NaN ──────────────────────────────────────
    # -1 means "word not found in lexicon" — not a real value.
    # SimpleImputer inside the pipeline will fill these with the training mean.
    feature_cols = [c for c in df.columns if c.startswith("f")]
    df[feature_cols] = df[feature_cols].replace(-1, np.nan)
    print(f"  {len(feature_cols)} feature columns found")

    # ── 3. One-hot encode categorical features ────────────────────────────────
    # Categories are defined explicitly so column order is always fixed
    # regardless of which values happen to appear in the data.
    #
    # f01_pos_code:     0=VERB, 1=NOUN, 2=ADJ, 3=ADV
    # f07_verbal_tense: 0=Pres, 1=Past, 2=Imp, 3=Fut  (-1=N/A already → NaN)
    #
    # One-hot is done at word level BEFORE aggregation so that the
    # aggregated mean = proportion of that category per subject.
    # e.g. f01_pos_code_0_mean = proportion of verbs used by this subject.

    POS_CATEGORIES   = [0, 1, 2, 3]          # VERB, NOUN, ADJ, ADV
    TENSE_CATEGORIES = [0, 1, 2, 3]          # Pres, Past, Imp, Fut

    if "f01_pos_code" in df.columns:
        df["f01_pos_code"] = pd.Categorical(
            df["f01_pos_code"], categories=POS_CATEGORIES
        )
        pos_dummies = pd.get_dummies(
            df["f01_pos_code"], prefix="f01_pos_code", dtype=float
        )
        # Enforce fixed column order
        pos_dummies = pos_dummies.reindex(
            columns=[f"f01_pos_code_{c}" for c in POS_CATEGORIES], fill_value=0.0
        )
        df = pd.concat([df.drop(columns=["f01_pos_code"]), pos_dummies], axis=1)
        print(f"  One-hot encoded f01_pos_code → {list(pos_dummies.columns)}")

    if "f07_verbal_tense" in df.columns:
        df["f07_verbal_tense"] = pd.Categorical(
            df["f07_verbal_tense"], categories=TENSE_CATEGORIES
        )
        tense_dummies = pd.get_dummies(
            df["f07_verbal_tense"], prefix="f07_verbal_tense", dtype=float
        )
        # Enforce fixed column order
        tense_dummies = tense_dummies.reindex(
            columns=[f"f07_verbal_tense_{c}" for c in TENSE_CATEGORIES], fill_value=0.0
        )
        df = pd.concat([df.drop(columns=["f07_verbal_tense"]), tense_dummies], axis=1)
        print(f"  One-hot encoded f07_verbal_tense → {list(tense_dummies.columns)}")

    # ── 4. Aggregate to static vector per subject ─────────────────────────────
    # Compute mean and std across all content words per subject.
    # mean → average value of the feature across the subject's words
    # std  → variability of the feature within the subject's words
    #
    # This gives 2 × n_features columns per subject.
    # Any column that is all-NaN for a subject stays NaN → imputed in pipeline.

    # Update feature_cols after one-hot encoding — numeric only
    feature_cols = [c for c in df.columns
                    if c.startswith("f")
                    and pd.api.types.is_numeric_dtype(df[c])
                    and c not in ("f01_pos_label",)]

    print("  Aggregating word-level features to subject-level static vectors...")
    agg = df.groupby("subject_id")[feature_cols].agg(["mean", "std"])
    agg.columns = [f"{feat}_{stat}" for feat, stat in agg.columns]
    agg = agg.reset_index()

    # Sort feature columns alphabetically so order is always deterministic
    # regardless of how columns were created (one-hot cols were appended last)
    feat_cols_sorted = sorted([c for c in agg.columns if c != "subject_id"])
    agg = agg[["subject_id"] + feat_cols_sorted]

    print(f"  Static matrix: {len(agg)} subjects × {len(feat_cols_sorted)} features")

    # ── 5. Extract labels ─────────────────────────────────────────────────────
    try:
        agg["label"] = agg["subject_id"].apply(extract_label)
    except ValueError as e:
        print(f"\n[error] {e}")
        print("Check that subject IDs start with 'HC' or 'PD'.")
        sys.exit(1)

    label_counts = agg["label"].value_counts().sort_index()
    print(f"  Labels: HC={label_counts.get(0, 0)}, PD={label_counts.get(1, 0)}")

    # ── 6. Build X, y, metadata ───────────────────────────────────────────────
    drop_cols          = ["subject_id", "label"]
    final_feature_cols = sorted(                       # alphabetical = fixed order
        [c for c in agg.columns if c not in drop_cols]
    )

    X             = agg[final_feature_cols].values.astype(np.float32)
    y             = agg["label"].values.astype(np.int64)
    subject_ids   = agg["subject_id"].values
    feature_names = np.array(final_feature_cols)

    print(f"\n  X shape        : {X.shape}")
    print(f"  y shape        : {y.shape}")
    print(f"  NaN per column : {np.isnan(X).sum(axis=0).max()} max across features")
    print(f"  First 5 features: {feature_names[:5].tolist()}")
    print(f"  Last  5 features: {feature_names[-5:].tolist()}")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    np.save(output_path / "X.npy",             X)
    np.save(output_path / "y.npy",             y)
    np.save(output_path / "subject_ids.npy",   subject_ids)
    np.save(output_path / "feature_names.npy", feature_names)

    # Also save a readable CSV for inspection
    agg.to_csv(output_path / "static_features.csv", index=False)

    print(f"\nSaved to {output_path}/")
    print(f"  X.npy              ← {X.shape}")
    print(f"  y.npy              ← {y.shape}")
    print(f"  subject_ids.npy    ← subject ID strings")
    print(f"  feature_names.npy  ← {len(feature_names)} feature names")
    print(f"  static_features.csv ← human-readable version")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare DisLanguage features for classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to all_subjects_features.csv from extract_features.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to save prepared arrays",
    )
    args = parser.parse_args()
    prepare(args.input, args.output)


if __name__ == "__main__":
    main()