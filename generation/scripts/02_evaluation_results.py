"""
jsd_evaluation.py
=================

Evaluates Jensen-Shannon Divergence between LLM-simulated GSS survey responses
and GSS microdata ground truth, split by model × persona file × variable.

Usage
-----
    python jsd_evaluation.py

Outputs (written to PROJECT_DIR/evaluation/):
    jsd_by_variable.csv          – JSD, TVD, homogenisation per model × persona × variable
    jsd_summary.csv              – Mean/median JSD per model × persona
    demographic_sensitivity.csv  – χ² sensitivity per model × demographic × variable
    jsd_heatmap.png
    homogenisation.png
"""

# ── 0. Configuration ──────────────────────────────────────────────────────────

from pathlib import Path

PROJECT_DIR = Path("/home/sant6886/ArtSoc/ArtificialSocieties/generation")

YEAR = 2024
SYNTHETIC_DIR = PROJECT_DIR / "synthetic_data" / f"year_2024"

# Stata microdata — actual GSS responses used as ground truth
GSS_DTA_PATH = PROJECT_DIR / "data" / "gss7224_r1.dta"

OUTPUT_DIR = PROJECT_DIR / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Demographic columns to test sensitivity against (must exist in persona CSVs)
DEMO_COLS = ["age", "sex", "race", "educ", "partyid", "polviews"]

# Persona file tag → path mapping (mirrors query_gss_comprehensive.py)
PERSONA_FILE_MAP = {
    "personas_demographics_political": PROJECT_DIR / "data" / "personas_demographics_political.csv",
    "personas_demographics"          : PROJECT_DIR / "data" / "personas_demographics.csv",
    "gss2024_personas"               : PROJECT_DIR / "data" / "gss2024_personas.csv",
}

# ── 1. Imports ────────────────────────────────────────────────────────────────

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for server/script use
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, entropy as scipy_entropy
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.4f}".format)

# ── 2. Core metric functions ──────────────────────────────────────────────────

def normalise(series: pd.Series, options: list) -> np.ndarray:
    """Probability vector over `options`; handles missing categories."""
    counts = series.value_counts().reindex(options, fill_value=0)
    total  = counts.sum()
    return (counts / total).values if total > 0 else np.ones(len(options)) / len(options)


def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence (base-2, range 0–1).
    scipy.jensenshannon returns the square-root (JS distance),
    so we square it to recover the proper divergence.
    """
    return float(jensenshannon(p, q, base=2) ** 2)


def compute_tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total Variation Distance."""
    return float(0.5 * np.sum(np.abs(p - q)))


def homogenisation_score(llm_series: pd.Series,
                          gss_series: pd.Series,
                          options: list) -> dict:
    """
    Compare Shannon entropy of LLM vs GSS distributions.
    ratio < 1  →  LLM responses are less diverse than real respondents.
    """
    p    = normalise(llm_series, options)
    q    = normalise(gss_series, options)
    H_llm = float(scipy_entropy(p + 1e-12, base=2))
    H_gss = float(scipy_entropy(q + 1e-12, base=2))
    return {
        "entropy_llm"          : H_llm,
        "entropy_gss"          : H_gss,
        "homogenisation_ratio" : H_llm / H_gss if H_gss > 0 else np.nan,
    }


def per_category_jsd_contribution(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-category additive contributions to JSD (sum ≈ JSD)."""
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.where(p > 0, p * np.log2(p / m), 0.0)
        kl_q = np.where(q > 0, q * np.log2(q / m), 0.0)
    return np.maximum(0, 0.5 * kl_p + 0.5 * kl_q)


def demographic_sensitivity(df: pd.DataFrame,
                             variable: str,
                             demo_col: str) -> dict:
    """Chi-squared test: do LLM responses vary across demographic groups?"""
    sub = df[df["variable"] == variable].dropna(subset=["answer", demo_col])
    if sub[demo_col].nunique() < 2 or len(sub) < 10:
        return {"chi2": np.nan, "p_value": np.nan, "sensitive": np.nan}
    ct = pd.crosstab(sub[demo_col], sub["answer"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"chi2": np.nan, "p_value": np.nan, "sensitive": np.nan}
    chi2, p, *_ = chi2_contingency(ct)
    return {"chi2": float(chi2), "p_value": float(p), "sensitive": p < 0.05}


# ── 3. Load synthetic CSV files ───────────────────────────────────────────────

print(f"\n{'='*65}")
print("ArtificialSocieties — JSD Evaluation")
print(f"{'='*65}")
print(f"Synthetic data : {SYNTHETIC_DIR}")
print(f"Ground truth   : {GSS_DTA_PATH}")
print(f"Output dir     : {OUTPUT_DIR}\n")

csv_files = sorted(SYNTHETIC_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(
        f"No CSVs found in {SYNTHETIC_DIR}. Check SYNTHETIC_DIR in the config."
    )

print(f"Found {len(csv_files)} CSV file(s):")
for f in csv_files:
    print(f"  {f.name}")

dfs = []
for f in csv_files:
    df = pd.read_csv(f, low_memory=False)
    df["source_file"] = f.name
    dfs.append(df)

raw = pd.concat(dfs, ignore_index=True)

# Keep only valid (answered, no error) rows
raw = raw[raw["answer"].notna() & (raw["error"].fillna("") == "")].copy()
raw["answer"] = raw["answer"].astype(str).str.strip()

print(f"\nTotal valid rows : {len(raw):,}")
print(f"Models           : {sorted(raw['model'].unique())}")
print(f"Persona files    : {sorted(raw['persona_file'].unique())}")
print(f"Variables        : {raw['variable'].nunique()}")


# ── 4. Load GSS ground truth (.dta) ──────────────────────────────────────────

if GSS_DTA_PATH.exists():
    print(f"\nLoading GSS microdata from {GSS_DTA_PATH.name} …")
    gss_actual = pd.read_stata(GSS_DTA_PATH, convert_categoricals=False)

    # Convert value-labelled columns manually, skipping any with duplicate labels
    reader = pd.io.stata.StataReader(GSS_DTA_PATH)
    value_labels = reader.value_labels()
    for col, label_map in value_labels.items():
        if col not in gss_actual.columns:
            continue
        # Only apply if labels are unique (no duplicates)
        if len(set(label_map.values())) == len(label_map):
            gss_actual[col] = gss_actual[col].map(label_map).fillna(gss_actual[col])

    # Normalise all string columns
    for col in gss_actual.select_dtypes(include=["object", "category"]).columns:
        gss_actual[col] = gss_actual[col].astype(str).str.strip()

    # Drop pure metadata/weight columns that are not survey variables
    # (keeps only columns that share names with variables in the synthetic data)
    shared_vars = [v for v in raw["variable"].unique() if v in gss_actual.columns]
    print(f"GSS rows         : {len(gss_actual):,}")
    print(f"Shared variables : {len(shared_vars)}")
    HAS_GROUND_TRUTH = len(shared_vars) > 0

    if not HAS_GROUND_TRUTH:
        print(
            "\n⚠  WARNING: No column names overlap between the .dta file and the\n"
            "   variable names in your synthetic CSVs. Check that the variable\n"
            "   names in GSS_QUESTIONS_COMPREHENSIVE match the .dta column names.\n"
            "   JSD vs. ground truth will be skipped.\n"
        )
else:
    gss_actual       = None
    HAS_GROUND_TRUTH = False
    shared_vars      = []
    print(f"\n⚠  GSS .dta not found at {GSS_DTA_PATH}. JSD vs. GSS will be skipped.")


# ── 5. Infer answer option sets per variable ──────────────────────────────────

sys.path.insert(0, str(PROJECT_DIR))

try:
    from generation.query_gss_comprehensive import GSS_QUESTIONS_COMPREHENSIVE   # type: ignore
    variable_options = {
        var: list(meta["options"])
        for var, meta in GSS_QUESTIONS_COMPREHENSIVE.items()
    }
    print(f"\nLoaded option sets for {len(variable_options)} variables from repo.")
except ImportError:
    print("\nCould not import GSS_QUESTIONS_COMPREHENSIVE — inferring options from data.")
    variable_options = {}
    for var in raw["variable"].unique():
        opts = set(raw[raw["variable"] == var]["answer"].dropna().unique())
        if HAS_GROUND_TRUTH and var in gss_actual.columns:
            opts |= set(gss_actual[var].dropna().astype(str).str.strip().unique())
        variable_options[var] = sorted(opts)


# ── 6. Compute JSD — model × persona file × variable ─────────────────────────

print(f"\n{'─'*65}")
print("Computing JSD …")
print(f"{'─'*65}")

results = []

for (model, persona_file), group_df in raw.groupby(["model", "persona_file"]):
    short_model = model.split("/")[-1]
    print(f"  {short_model}  |  {persona_file}  ({len(group_df):,} rows)")

    for var in group_df["variable"].unique():
        options     = variable_options.get(var)
        llm_answers = group_df[group_df["variable"] == var]["answer"].dropna()

        if not options or len(llm_answers) < 5:
            continue

        p_llm = normalise(llm_answers, options)

        row = {
            "model"           : model,
            "persona_file"    : persona_file,
            "variable"        : var,
            "n_llm_responses" : len(llm_answers),
        }

        if HAS_GROUND_TRUTH and var in gss_actual.columns:
            gss_answers = gss_actual[var].dropna().astype(str).str.strip()
            p_gss       = normalise(gss_answers, options)
            jsd_val     = compute_jsd(p_llm, p_gss)
            contrib     = per_category_jsd_contribution(p_llm, p_gss)
            homo        = homogenisation_score(llm_answers, gss_answers, options)

            row.update({
                "n_gss_responses"         : len(gss_answers),
                "jsd"                     : jsd_val,
                "jsd_sqrt"                : float(np.sqrt(jsd_val)),
                "tvd"                     : compute_tvd(p_llm, p_gss),
                "entropy_llm"             : homo["entropy_llm"],
                "entropy_gss"             : homo["entropy_gss"],
                "homogenisation_ratio"    : homo["homogenisation_ratio"],
                "worst_category"          : options[int(np.argmax(contrib))],
                "worst_category_contrib"  : float(np.max(contrib)),
            })
        else:
            row.update({
                "n_gss_responses"         : np.nan,
                "jsd"                     : np.nan,
                "jsd_sqrt"                : np.nan,
                "tvd"                     : np.nan,
                "entropy_llm"             : float(scipy_entropy(p_llm + 1e-12, base=2)),
                "entropy_gss"             : np.nan,
                "homogenisation_ratio"    : np.nan,
                "worst_category"          : np.nan,
                "worst_category_contrib"  : np.nan,
            })

        results.append(row)

results_df = pd.DataFrame(results)
print(f"\nEvaluation rows computed: {len(results_df):,}")


# ── 7. Summary table ──────────────────────────────────────────────────────────

summary = (
    results_df
    .groupby(["model", "persona_file"])
    .agg(
        n_variables              = ("variable",               "count"),
        mean_jsd                 = ("jsd",                    "mean"),
        median_jsd               = ("jsd",                    "median"),
        mean_jsd_sqrt            = ("jsd_sqrt",               "mean"),
        mean_tvd                 = ("tvd",                    "mean"),
        mean_homogenisation_ratio= ("homogenisation_ratio",   "mean"),
        pct_vars_jsd_gt_015      = ("jsd", lambda x: (x > 0.15).mean() * 100),
    )
    .reset_index()
    .sort_values("mean_jsd")
)

print(f"\n{'─'*65}")
print("Summary — mean JSD per model × persona file")
print(f"{'─'*65}")
print(summary.to_string(index=False))


# ── 8. Worst variables per model ──────────────────────────────────────────────

if HAS_GROUND_TRUTH:
    print(f"\n{'─'*65}")
    print("Top 10 worst variables (highest JSD) per model × persona file")
    print(f"{'─'*65}")
    for (model, pfile), grp in results_df.groupby(["model", "persona_file"]):
        print(f"\n  {model.split('/')[-1]}  |  {pfile}")
        top = grp.nlargest(10, "jsd")[
            ["variable", "jsd", "tvd", "homogenisation_ratio", "worst_category"]
        ]
        print(top.to_string(index=False))
else:
    print(f"\n{'─'*65}")
    print("Variables by LLM entropy (lowest = most homogenised)")
    print(f"{'─'*65}")
    for (model, pfile), grp in results_df.groupby(["model", "persona_file"]):
        print(f"\n  {model.split('/')[-1]}  |  {pfile}")
        print(grp.nsmallest(10, "entropy_llm")[["variable", "entropy_llm"]].to_string(index=False))

# ── 9. Heatmap ────────────────────────────────────────────────────────────────

if HAS_GROUND_TRUTH and results_df["jsd"].notna().any():
    pivot = results_df.pivot_table(
        index="variable", columns=["model", "persona_file"],
        values="jsd", aggfunc="mean",
    )
    pivot.columns = [" | ".join(c).strip() for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2.5),
                                    max(8, len(pivot) * 0.35)))
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn_r", vmin=0, vmax=0.5,
                annot=True, fmt=".3f", linewidths=0.4,
                cbar_kws={"label": "JSD  (0 = identical, 1 = completely different)"})
    ax.set_title("JSD by variable × model | persona file", fontsize=13, pad=12)
    ax.set_xlabel(""); ax.set_ylabel("GSS variable")
    plt.tight_layout()
    out = OUTPUT_DIR / "jsd_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved heatmap → {out}")


# ── 9b. Bar charts — JSD and homogenisation by model and by persona file ──────

def _ci95(series):
    """95% confidence interval half-width using t-distribution."""
    from scipy.stats import t as t_dist
    s = series.dropna()
    if len(s) < 2:
        return 0.0
    return float(t_dist.ppf(0.975, df=len(s) - 1) * s.std(ddof=1) / np.sqrt(len(s)))


if HAS_GROUND_TRUTH and results_df["jsd"].notna().any():

    MODEL_LABELS = {m: m.split("/")[-1] for m in results_df["model"].unique()}
    PERSONA_LABELS = {
        "gss2024_personas"               : "GSS 2024\nPersonas",
        "personas_demographics"          : "Demographics\nOnly",
        "personas_demographics_political": "Demographics\n+ Political",
    }

    COLORS_MODEL   = ["#378ADD", "#D85A30", "#1D9E75"]
    COLORS_PERSONA = ["#7F77DD", "#BA7517", "#0F6E56"]

    # ── pool by MODEL ─────────────────────────────────────────────────────────
    by_model_jsd  = results_df.groupby("model")["jsd"].agg(["mean", _ci95]).reset_index()
    by_model_homo = results_df.groupby("model")["homogenisation_ratio"].agg(["mean", _ci95]).reset_index()

    # ── pool by PERSONA FILE ──────────────────────────────────────────────────
    by_pfile_jsd  = results_df.groupby("persona_file")["jsd"].agg(["mean", _ci95]).reset_index()
    by_pfile_homo = results_df.groupby("persona_file")["homogenisation_ratio"].agg(["mean", _ci95]).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LLM Survey Alignment — Pooled across variables", fontsize=14, y=1.01)

    def _bar_plot(ax, data, group_col, val_col, ci_col, labels_map, colors,
                  title, ylabel, ref_line=None):
        xs     = np.arange(len(data))
        labels = [labels_map.get(v, v) for v in data[group_col]]
        bars   = ax.bar(xs, data[val_col], yerr=data[ci_col],
                        color=colors[:len(xs)], width=0.55,
                        capsize=6, error_kw={"linewidth": 1.5, "ecolor": "black"},
                        zorder=3)
        if ref_line is not None:
            ax.axhline(ref_line, color="black", linestyle="--", linewidth=1,
                       label=f"Reference = {ref_line}", zorder=4)
            ax.legend(fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, pad=8)
        ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        # Annotate bar tops
        for bar, ci in zip(bars, data[ci_col]):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + ci + 0.004,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    # Top-left: JSD by model
    _bar_plot(axes[0, 0], by_model_jsd, "model", "mean", "_ci95",
              MODEL_LABELS, COLORS_MODEL,
              "Mean JSD by model\n(pooled across variables & persona files)",
              "JSD  (lower = better)")

    # Top-right: JSD by persona file
    _bar_plot(axes[0, 1], by_pfile_jsd, "persona_file", "mean", "_ci95",
              PERSONA_LABELS, COLORS_PERSONA,
              "Mean JSD by persona file\n(pooled across variables & models)",
              "JSD  (lower = better)")

    # Bottom-left: homogenisation by model
    _bar_plot(axes[1, 0], by_model_homo, "model", "mean", "_ci95",
              MODEL_LABELS, COLORS_MODEL,
              "Mean homogenisation ratio by model\n(pooled across variables & persona files)",
              "Entropy ratio  (1 = same diversity as GSS)",
              ref_line=1.0)

    # Bottom-right: homogenisation by persona file
    _bar_plot(axes[1, 1], by_pfile_homo, "persona_file", "mean", "_ci95",
              PERSONA_LABELS, COLORS_PERSONA,
              "Mean homogenisation ratio by persona file\n(pooled across variables & models)",
              "Entropy ratio  (1 = same diversity as GSS)",
              ref_line=1.0)

    plt.tight_layout()
    out = OUTPUT_DIR / "jsd_homogenisation_bars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved bar charts → {out}")


# ── 10. Homogenisation KDE plot ───────────────────────────────────────────────