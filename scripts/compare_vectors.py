"""
compare_vectors.py
==================
Compare feature outputs extracted with different word vector models
(e.g. fastText vs Word2Vec).

Only compares the vector-dependent features:
    f11_local_sem_var, f12_global_sem_var  (Lexical)
    f21_dist_manipulation, f22_dist_motor_action, f23_dist_abstract  (Semantic)

Usage
-----
    python scripts/compare_vectors.py \\
        --a  data/PC-GITA/features_fasttext/all_subjects_features.csv \\
        --b  data/PC-GITA/features_w2v/all_subjects_features.csv \\
        --label-a fastText \\
        --label-b Word2Vec
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ── Features that depend on word vectors ─────────────────────────────────────

VECTOR_FEATURES = [
    "f11_local_sem_var",
    "f12_global_sem_var",
    "f21_dist_manipulation",
    "f22_dist_motor_action",
    "f23_dist_abstract",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.replace(-1, np.nan)
    return df


def coverage(df: pd.DataFrame, col: str) -> float:
    return df[col].notna().mean() * 100


def stats(series: pd.Series) -> dict:
    s = series.dropna()
    return {
        "n":    len(s),
        "mean": round(s.mean(), 4),
        "std":  round(s.std(),  4),
        "min":  round(s.min(),  4),
        "max":  round(s.max(),  4),
    }


def correlation(a: pd.Series, b: pd.Series) -> float:
    """Pearson correlation between two series, aligned by index."""
    both = pd.concat([a, b], axis=1).dropna()
    if len(both) < 2:
        return float("nan")
    return round(both.iloc[:, 0].corr(both.iloc[:, 1]), 4)


def mean_abs_diff(a: pd.Series, b: pd.Series) -> float:
    both = pd.concat([a, b], axis=1).dropna()
    if both.empty:
        return float("nan")
    return round((both.iloc[:, 0] - both.iloc[:, 1]).abs().mean(), 4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Compare vector-dependent features across two extraction runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--a",       required=True, help="Path to first all_subjects_features.csv")
    p.add_argument("--b",       required=True, help="Path to second all_subjects_features.csv")
    p.add_argument("--label-a", default="Model A", help="Label for first model")
    p.add_argument("--label-b", default="Model B", help="Label for second model")
    p.add_argument("--output",  default=None,
                   help="Optional path to save comparison CSV")
    args = p.parse_args()

    print(f"\nLoading features...")
    df_a = load(args.a)
    df_b = load(args.b)
    print(f"  {args.label_a}: {len(df_a):,} word rows, {df_a['subject_id'].nunique()} subjects")
    print(f"  {args.label_b}: {len(df_b):,} word rows, {df_b['subject_id'].nunique()} subjects")

    # Align on subject_id + form + lemma (same word, same position)
    merge_cols = ["subject_id", "form", "lemma"]
    merged = df_a[merge_cols + VECTOR_FEATURES].merge(
        df_b[merge_cols + VECTOR_FEATURES],
        on=merge_cols,
        suffixes=(f"_{args.label_a}", f"_{args.label_b}"),
    )
    print(f"\n  Aligned rows: {len(merged):,}")

    # ── Per-feature comparison ─────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {'Feature':<28} {'Cov-A':>6} {'Cov-B':>6} {'Corr':>8} {'MAD':>8}")
    print(f"{'='*70}")

    rows = []
    for feat in VECTOR_FEATURES:
        col_a = f"{feat}_{args.label_a}"
        col_b = f"{feat}_{args.label_b}"

        if col_a not in merged.columns or col_b not in merged.columns:
            print(f"  {feat:<28} — feature not found in one of the files")
            continue

        cov_a = coverage(df_a, feat)
        cov_b = coverage(df_b, feat)
        corr  = correlation(merged[col_a], merged[col_b])
        mad   = mean_abs_diff(merged[col_a], merged[col_b])

        print(f"  {feat:<28} {cov_a:>5.1f}% {cov_b:>5.1f}% {corr:>8.4f} {mad:>8.4f}")

        rows.append({
            "feature":              feat,
            f"coverage_{args.label_a}": round(cov_a, 2),
            f"coverage_{args.label_b}": round(cov_b, 2),
            "pearson_correlation":  corr,
            "mean_abs_diff":        mad,
        })

    print(f"{'='*70}")
    print(f"  Corr = Pearson correlation (1.0 = identical rankings)")
    print(f"  MAD  = Mean absolute difference (lower = more similar values)")

    # ── Distribution comparison per feature ───────────────────
    print(f"\n{'='*70}")
    print("  Per-feature distributions")
    print(f"{'='*70}")

    for feat in VECTOR_FEATURES:
        col_a = f"{feat}_{args.label_a}"
        col_b = f"{feat}_{args.label_b}"
        if col_a not in merged.columns:
            continue

        s_a = stats(merged[col_a])
        s_b = stats(merged[col_b])

        print(f"\n  {feat}")
        print(f"    {'':20} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
        print(f"    {args.label_a:<20} {s_a['mean']:>8} {s_a['std']:>8} {s_a['min']:>8} {s_a['max']:>8}")
        print(f"    {args.label_b:<20} {s_b['mean']:>8} {s_b['std']:>8} {s_b['min']:>8} {s_b['max']:>8}")

    # ── OOV comparison ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  OOV (out-of-vocabulary) comparison — words missing in each model")
    print(f"{'='*70}")

    for feat in VECTOR_FEATURES[:1]:  # same OOV pattern for all vector features
        oov_a = df_a[feat].isna()
        oov_b = df_b[feat].isna()
        only_a_oov = (~oov_a & oov_b)   # found in A, missing in B
        only_b_oov = (oov_a & ~oov_b)   # found in B, missing in A
        both_oov   = (oov_a & oov_b)

        print(f"\n  Based on {feat}:")
        print(f"    Found in {args.label_a} only : {only_b_oov.sum():,} words")
        print(f"    Found in {args.label_b} only : {only_a_oov.sum():,} words")
        print(f"    Missing in both              : {both_oov.sum():,} words")

        if only_a_oov.sum() > 0:
            examples = df_b.loc[only_a_oov, "lemma"].value_counts().head(10)
            print(f"\n    Words found in {args.label_b} but not {args.label_a} (top 10):")
            for lemma, count in examples.items():
                print(f"      {lemma} ({count}×)")

        if only_b_oov.sum() > 0:
            examples = df_a.loc[only_b_oov, "lemma"].value_counts().head(10)
            print(f"\n    Words found in {args.label_a} but not {args.label_b} (top 10):")
            for lemma, count in examples.items():
                print(f"      {lemma} ({count}×)")

    # ── Save summary CSV ──────────────────────────────────────
    if rows:
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = Path(args.a).parent.parent / "comparison_vectors.csv"

        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\n  Summary saved to {out_path}")


if __name__ == "__main__":
    main()