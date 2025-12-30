

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from scipy.stats import ttest_ind
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.lines import Line2D

WD = Path("/media/jie/expand_5t/7exp_expand/tang_all/tang_v2/GaussDiff/CHARLS")
SEED_GLOBAL = 42
N_ROW = 6
N_COL = 20

if not sys.warnoptions:
    warnings.simplefilter("ignore")

np.random.seed(SEED_GLOBAL)
pd.set_option("display.max_columns", N_COL)
pd.set_option("display.min_rows", N_ROW)

print("current:", os.getcwd())
os.chdir(WD)
print("defined:", str(WD))

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def remove_if_exists(file_path: Path) -> None:
    if file_path.exists() and file_path.is_file():
        file_path.unlink()
    print(f"remove_create: {file_path}")

def cut_value_counts(data: pd.Series, bins, verbose: bool = False) -> pd.DataFrame:
    data_binned = pd.cut(data, bins=bins)
    counts = data_binned.value_counts().reset_index()
    counts.columns = ["bin", "count"]
    counts = counts.sort_values(by="bin")
    if verbose:
        print(counts)
    return counts

def require_file(p: Path, hint: str = "") -> None:
    if not p.exists():
        msg = f"Missing file: {p}"
        if hint:
            msg += f"\nHint: {hint}"
        raise FileNotFoundError(msg)

def _clean_finite(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X

METRICS_CSV = Path("results/eval/metrics_all.csv")
require_file(METRICS_CSV, "Run from the project root where results/eval/metrics_all.csv exists.")
df_met = pd.read_csv(METRICS_CSV)

TYPE_ALIAS = {
    "Random_Oversampling": "Random oversampling",
    "Random oversampling": "Random oversampling",
}
TYPE_COL = "type"
if TYPE_COL in df_met.columns:
    df_met[TYPE_COL] = df_met[TYPE_COL].replace(TYPE_ALIAS)

PLOT_METRICS_DIR = ensure_dir(Path("plot/metrics"))

ROW_SPECS = [
    ("GaussDiff", "GaussDiff", "global"),
    ("Baseline",  "TVAE", "TVAE"),
    ("Baseline",  "CTGAN", "CTGAN"),
    ("Baseline",  "SVMSMOTE", "SVMSMOTE"),
    ("Baseline",  "SMOTE", "SMOTE"),
    ("Baseline",  "ADASYN", "ADASYN"),
    ("Baseline",  "Random oversampling", "Random oversampling"),
]

METRIC_BLOCKS = [
    ("recall", "Recall"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
]

B_BOOT = 10000
ALPHA = 0.05
SEED_BOOT = 2025
rng_boot = np.random.default_rng(SEED_BOOT)

def bootstrap_ci_mean(x: np.ndarray, rng, B: int = 10000, alpha: float = 0.05):
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0

    if n == 1:
        return mu, sd, np.nan, np.nan, n

    idx = rng.integers(0, n, size=(B, n))
    boot_means = x[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return mu, sd, lo, hi, n

def fmt_mean_sd(mu, sd, nd=3):
    if not np.isfinite(mu):
        return "NA"
    if not np.isfinite(sd):
        return f"{mu:.{nd}f}"
    return f"{mu:.{nd}f} ± {sd:.{nd}f}"

def fmt_ci(lo, hi, nd=3):
    if np.isfinite(lo) and np.isfinite(hi):
        return f"[{lo:.{nd}f}, {hi:.{nd}f}]"
    return "NA"

present_types = set(df_met[TYPE_COL].dropna().unique())

boot_stats = {}
for t in present_types:
    g = df_met.loc[df_met[TYPE_COL] == t]
    for m, _ in METRIC_BLOCKS:
        if m not in g.columns:
            raise ValueError(f"Missing column in metrics_all.csv: {m}")
        x = g[m].to_numpy(dtype=float)
        mu, sd, lo, hi, n = bootstrap_ci_mean(x, rng=rng_boot, B=B_BOOT, alpha=ALPHA)
        boot_stats[(t, m)] = (mu, sd, lo, hi, n)

rows = []
for group, t, display_name in ROW_SPECS:
    if t not in present_types:
        continue
    row = {"Group": group, "": display_name}
    for m, mdisp in METRIC_BLOCKS:
        mu, sd, lo, hi, n = boot_stats[(t, m)]
        row[(mdisp, "mean ± sd")] = fmt_mean_sd(mu, sd, nd=3)
        row[(mdisp, "95% confidence interval")] = fmt_ci(lo, hi, nd=3)
    rows.append(row)

df_out = pd.DataFrame(rows)
left_cols = ["Group", ""]
metric_cols = [c for c in df_out.columns if isinstance(c, tuple)]
df_out = df_out[left_cols + metric_cols]
df_out.loc[df_out["Group"].duplicated(), "Group"] = ""

OUT_CSV = PLOT_METRICS_DIR / "metrics_summary_bootstrap_ci95_wide.csv"
df_out.to_csv(OUT_CSV, index=False)

print(df_out.to_string(index=False))
print("\nsaved:", OUT_CSV)
print(f"\nBootstrap settings: B={B_BOOT}, alpha={ALPHA}, seed={SEED_BOOT}")

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.size"] = 8.5
mpl.rcParams["axes.titlesize"] = 8.0
mpl.rcParams["axes.labelsize"] = 8.5
mpl.rcParams["xtick.labelsize"] = 8.0
mpl.rcParams["ytick.labelsize"] = 7.0
mpl.rcParams["text.usetex"] = False
sns.set(style="white")

ORDER = [
    "GaussDiff",
    "TVAE",
    "CTGAN",
    "ADASYN",
    "SMOTE",
    "Random oversampling",
]
order_use = [t for t in ORDER if t in present_types]

warm = sns.color_palette("Reds", 6)[2:5]
cool = sns.color_palette("Set2", 10)
palette_map = {
    "GaussDiff": warm[2],
    "TVAE": cool[2],
    "CTGAN": cool[4],
    "ADASYN": cool[6],
    "SMOTE": cool[7],
    "Random oversampling": cool[8],
}
for t in order_use:
    palette_map.setdefault(t, cool[0])
palette_list = [palette_map[t] for t in order_use]

METRICS_LIST = ["pr_auc", "roc_auc", "recall"]
Y_LABELS = ["PR-AUC(→)", "ROC-AUC(→)", "Recall(→)"]

df_melt = (
    df_met.melt(
        id_vars=[TYPE_COL],
        value_vars=METRICS_LIST,
        var_name="metrics",
        value_name="value",
    )
    .rename(columns={TYPE_COL: "model"})
)

def welch_p(a, b):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return ttest_ind(a, b, equal_var=False).pvalue

def fmt_p(p):
    if not np.isfinite(p):
        return "NA"
    if p < 1e-3:
        exp = int(np.floor(np.log10(p)))
        coef = p / (10 ** exp)
        return f"{coef:.1f}e{exp}"
    return f"{p:.4f}"

fig_w = 3.7 / 2.54
fig_h = 16.0 / 2.54
fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h))
plt.subplots_adjust(hspace=0.06)

print("\n=== Welch t-test p-values ===")

for i, (ax, metric) in enumerate(zip(axes, METRICS_LIST)):
    df_metric = df_melt[df_melt["metrics"] == metric]
    metric_summary = df_metric.groupby("model")["value"].agg(["mean", "std"]).reindex(order_use)

    sns.barplot(
        x="model",
        y="value",
        data=df_metric,
        order=order_use,
        palette=palette_list,
        ci="sd",
        capsize=0.12,
        errwidth=0.9,
        ax=ax,
    )

    ax.set_ylabel(Y_LABELS[i], fontsize=8.0)
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_right()

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", labelrotation=90, labelsize=7)

    ax.set_xlabel("")
    ax.set_xticklabels([])
    if i == 2:
        ax.set_xticklabels(order_use, rotation=90, ha="center", va="top", fontsize=8.0)

    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    y0, y1 = ax.get_ylim()
    span = y1 - y0
    ax.set_ylim(y0, y1 + 0.22 * span)

    y0, y1 = ax.get_ylim()
    span = y1 - y0
    y_base = y0 + 0.03 * span
    y_up = 0.1 * span

    for j, model in enumerate(order_use):
        mu = metric_summary.loc[model, "mean"]
        sdv = metric_summary.loc[model, "std"]
        if not np.isfinite(mu):
            continue
        txt = f"{mu:.3f}±{sdv:.3f}"
        move_up = (metric == "recall" and mu < 0.3)
        ax.text(
            j,
            (mu + y_up) if move_up else y_base,
            txt,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8.5 * (0.80 if move_up else 0.82),
            color="black" if move_up else "white",
        )

    gauss = df_metric[df_metric["model"] == "GaussDiff"]["value"].to_numpy()
    baselines = [m for m in order_use if not m.startswith("GaussDiff")]
    if len(baselines) > 0:
        best_base = (
            df_metric[df_metric["model"].isin(baselines)]
            .groupby("model")["value"]
            .mean()
            .idxmax()
        )
        base_vals = df_metric[df_metric["model"] == best_base]["value"].to_numpy()
        p = welch_p(gauss, base_vals)
        print(f"{metric:8s} | GaussDiff vs {best_base:<22s} | p = {fmt_p(p)}")

OUT_METRICS_PNG = PLOT_METRICS_DIR / "metrics.png"
fig.savefig(OUT_METRICS_PNG, dpi=600, bbox_inches="tight")
plt.close(fig)
print("\nsaved:", OUT_METRICS_PNG)

PLOT_KM_DIR = ensure_dir(Path("plot/km"))

y_true = pd.read_csv("data/2_split/y_test.csv", index_col=0)
df_duration = pd.read_csv("data/4_km/km_event_year.csv", index_col=0)

n_eval_samples = len(sorted(glob("results/eval/sample_*/")))
if n_eval_samples == 0:
    raise FileNotFoundError("Missing directory: results/eval/sample_*/")

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.titlesize"] = 8
mpl.rcParams["axes.labelsize"] = 8
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["legend.fontsize"] = 8
mpl.rcParams["text.usetex"] = False

def _format_pvalue_with_stars(p: float):
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        stars = ""

    if p < 0.01:
        s = f"{p:.0e}"
        base, exp = s.split("e")
        base = float(base)
        exp = int(exp)
        p_fmt = rf"${base:g} \times 10^{{{exp}}}$"
    else:
        p_fmt = f"{p:.2f}"
    return stars, p_fmt

def plot_km_2x4_tight(
    df_km2,
    model_order,
    fig_w_cm=16.0,
    fig_h_cm=6.0,
    lw=1.0,
    p_text_x=0.03,
    p_text_y=0.10,
    wspace=0.09,
    hspace=0.14,
    xtick_pad=0.2,
    ytick_pad=0.2,
    xlabel_pad=0.8,
    ylabel_pad=0.8,
):
    df = df_km2.copy()
    df = df.loc[df["time"].notnull(), :].copy()
    df["events"] = pd.to_numeric(df["events"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.loc[df["events"].notnull() & df["time"].notnull(), :].copy()

    inch = 1 / 2.54
    fig, axes = plt.subplots(2, 4, figsize=(fig_w_cm * inch, fig_h_cm * inch))
    axes = np.array(axes).ravel()

    kmf = KaplanMeierFitter()

    c0 = mpl.rcParams["axes.prop_cycle"].by_key()["color"][0]
    c1 = mpl.rcParams["axes.prop_cycle"].by_key()["color"][1]
    legend_handles = [
        Line2D([0], [0], color=c0, lw=lw, label="predicted non-CVD"),
        Line2D([0], [0], color=c1, lw=lw, label="predicted CVD"),
    ]

    plot_positions = [0, 1, 2, 4, 5, 6]
    legend_pos = 3
    blank_pos = 7

    axes[legend_pos].axis("off")
    axes[blank_pos].axis("off")

    for i, model_name in enumerate(model_order):
        ax_idx = plot_positions[i]
        ax = axes[ax_idx]
        row = ax_idx // 4
        col = ax_idx % 4

        g = pd.to_numeric(df[model_name], errors="coerce")
        group = (g >= 0.5).astype(int)

        tmp = df[["time", "events"]].copy()
        tmp["group"] = group.values
        tmp = tmp.loc[tmp["group"].notnull(), :]

        group_A = tmp[tmp["group"] == 0]
        group_B = tmp[tmp["group"] == 1]

        if (len(group_A) == 0) or (len(group_B) == 0):
            kmf.fit(tmp["time"], event_observed=tmp["events"], label="predicted non-CVD")
            kmf.plot(ax=ax, linewidth=lw)
        else:
            kmf.fit(group_A["time"], event_observed=group_A["events"], label="predicted non-CVD")
            kmf.plot(ax=ax, linewidth=lw)
            kmf.fit(group_B["time"], event_observed=group_B["events"], label="predicted CVD")
            kmf.plot(ax=ax, linewidth=lw)

            res = logrank_test(
                group_A["time"],
                group_B["time"],
                event_observed_A=group_A["events"],
                event_observed_B=group_B["events"],
            )
            p = float(res.p_value)
            stars, p_fmt = _format_pvalue_with_stars(p)
            ax.text(
                p_text_x,
                p_text_y,
                f"{stars} p-value={p_fmt}",
                transform=ax.transAxes,
                fontsize=8,
                color="red",
            )

        ax.set_title(model_name, pad=1.5)
        ax.set_ylim(0.40, 1.02)

        ax.tick_params(axis="x", pad=xtick_pad)
        ax.tick_params(axis="y", pad=ytick_pad)

        if row == 0:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        else:
            ax.set_xlabel("Year", labelpad=xlabel_pad)

        if col == 0:
            ax.set_ylabel("Probabilities", labelpad=ylabel_pad)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    ax_leg = axes[legend_pos]
    ax_leg.axis("off")
    ax_leg.legend(handles=legend_handles, frameon=False, loc="center left")

    fig.subplots_adjust(
        left=0.045,
        right=0.995,
        bottom=0.10,
        top=0.92,
        wspace=wspace,
        hspace=hspace,
    )
    return fig, axes

PROBA_SPECS = [
    ("Random oversampling", "0_Random_Oversampling"),
    ("SMOTE", "1_SMOTE"),
    ("ADASYN", "2_ADASYN"),
    ("TVAE", "3_TVAE"),
    ("CTGAN", "4_CTGAN"),
    ("GaussDiff", "5_GaussDiff"),
]

model_order = ["GaussDiff", "TVAE", "CTGAN", "ADASYN", "SMOTE", "Random oversampling"]

for sample_id_ in tqdm(range(n_eval_samples), desc="KM"):
    dfs = [y_true, df_duration]

    for disp_name, folder in PROBA_SPECS:
        p_path = Path(f"results/eval/sample_{sample_id_:02d}/{folder}/y_proba.csv")
        require_file(p_path, f"Missing y_proba.csv in eval sample {sample_id_:02d}.")
        d = pd.read_csv(p_path)
        d.columns = [disp_name]
        d.index = y_true.index
        dfs.append(d)

    df_km = pd.concat(dfs, axis=1)
    df_km2 = df_km.loc[df_km.notnull().all(axis=1), :].copy()

    if "CVD" in df_km2.columns:
        if ((df_km2["CVD"] != df_km2["events"]).sum() != 0):
            raise ValueError(f"Label mismatch between y_test and km_event_year for sample {sample_id_:02d}.")
        df_km2 = df_km2.drop("CVD", axis=1)

    fig, axes = plot_km_2x4_tight(
        df_km2=df_km2,
        model_order=model_order,
        fig_w_cm=16.0,
        fig_h_cm=6.0,
        lw=1.0,
        p_text_x=0.03,
        p_text_y=0.10,
        wspace=0.09,
        hspace=0.14,
        xtick_pad=0.2,
        ytick_pad=0.2,
        xlabel_pad=0.8,
        ylabel_pad=0.8,
    )
    fig.savefig(PLOT_KM_DIR / f"km_sample_{sample_id_:02d}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

print("KM done")

PLOT_DCR_DIR = ensure_dir(Path("plot/dcr"))

MAPPING = {"A": 0, "B": 1, "No": 0, "Yes": 1, "Male": 0, "Female": 1}

def create_real_dataset(split_set: str):
    real_data_dir = Path("data/2_split")
    X_num = pd.read_csv(real_data_dir / f"X_num_{split_set}.csv", index_col=0)
    X_cat = pd.read_csv(real_data_dir / f"X_cat_{split_set}.csv", index_col=0)
    y = pd.read_csv(real_data_dir / f"y_{split_set}.csv", index_col=0)
    X_all = pd.concat([X_num, X_cat], axis=1)
    return X_all, y["CVD"]

Xs_train, y_real_train = create_real_dataset("train")
Xs_test, y_real_test = create_real_dataset("test")

def _align_and_map_object_cols(X: pd.DataFrame, ref_cols) -> pd.DataFrame:
    X = X.reindex(columns=ref_cols)
    obj_cols = X.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        X[c] = X[c].map(MAPPING)
    return X

def read_generated_gauss(sample_id_: int):
    d = Path(f"results/sample/sample_{sample_id_:02d}")
    require_file(d / "X_reversed.csv", "GaussDiff sample is missing X_reversed.csv.")
    require_file(d / "y_.csv", "GaussDiff sample is missing y_.csv.")
    X_gen = pd.read_csv(d / "X_reversed.csv")
    y_gen = pd.read_csv(d / "y_.csv")["CVD"]
    X_gen = _align_and_map_object_cols(X_gen, Xs_test.columns)
    return X_gen, y_gen

def read_generated_baseline(sample_id_: int, which: str):
    if which == "TVAE":
        d = Path(f"results/sample_tvae/sample_{sample_id_:02d}")
    elif which == "CTGAN":
        d = Path(f"results/sample_ctgan/sample_{sample_id_:02d}")
    else:
        raise ValueError(which)

    require_file(d / "X_.csv", f"{which} sample is missing X_.csv.")
    require_file(d / "y_.csv", f"{which} sample is missing y_.csv.")

    X = pd.read_csv(d / "X_.csv")
    y = pd.read_csv(d / "y_.csv")["CVD"]
    X = X.reindex(columns=Xs_test.columns)

    obj_cols = X.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        X[c] = X[c].map(MAPPING)

    return X, y

def dcr_to_reference(X_query: pd.DataFrame, X_ref: pd.DataFrame, scaler: StandardScaler | None = None):
    X_ref = _clean_finite(X_ref).dropna(axis=0, how="any")
    X_query = _clean_finite(X_query).dropna(axis=0, how="any")

    if scaler is None:
        scaler = StandardScaler()
        X_ref_z = scaler.fit_transform(X_ref)
    else:
        X_ref_z = scaler.transform(X_ref)

    X_query_z = scaler.transform(X_query)
    if not np.isfinite(X_ref_z).all() or not np.isfinite(X_query_z).all():
        raise ValueError("Non-finite values remain after scaling.")

    tree = cKDTree(X_ref_z)
    dcr, _ = tree.query(X_query_z, k=1)
    return dcr, scaler

def r2r_baseline(X_train: pd.DataFrame, scaler: StandardScaler):
    X_train_clean = _clean_finite(X_train).dropna(axis=0, how="any")
    X_train_z = scaler.transform(X_train_clean)
    tree = cKDTree(X_train_z)
    d2, _ = tree.query(X_train_z, k=2)
    return d2[:, 1]

rng_pair = np.random.default_rng(42)

def sample_pair(X_syn: pd.DataFrame, X_test_: pd.DataFrame):
    n = min(len(X_syn), len(X_test_))
    idx_syn = rng_pair.choice(len(X_syn), size=n, replace=False)
    idx_test = rng_pair.choice(len(X_test_), size=n, replace=False)
    return X_syn.iloc[idx_syn], X_test_.iloc[idx_test]

def ecdf(x):
    x = np.sort(np.asarray(x))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

n_samples = len(sorted(glob("results/sample/sample_*/")))
if n_samples == 0:
    n_samples = len(sorted(glob("results/eval/sample_*/")))

print(f"\nDCR: detected samples = {n_samples}")

for sample_id_ in tqdm(range(n_samples), desc="DCR"):
    Xs_gauss, _ = read_generated_gauss(sample_id_)
    Xs_tvae, _ = read_generated_baseline(sample_id_, "TVAE")
    Xs_ctgan, _ = read_generated_baseline(sample_id_, "CTGAN")

    Xg_q, Xt_q = sample_pair(Xs_gauss, Xs_test)
    dcr_g_s2r, scaler_g = dcr_to_reference(Xg_q, Xs_train, scaler=None)
    dcr_g_t2r, _ = dcr_to_reference(Xt_q, Xs_train, scaler=scaler_g)
    dcr_g_r2r = r2r_baseline(Xs_train, scaler_g)

    Xv_q, Xt_q = sample_pair(Xs_tvae, Xs_test)
    dcr_v_s2r, scaler_v = dcr_to_reference(Xv_q, Xs_train, scaler=None)
    dcr_v_t2r, _ = dcr_to_reference(Xt_q, Xs_train, scaler=scaler_v)
    dcr_v_r2r = r2r_baseline(Xs_train, scaler_v)

    Xc_q, Xt_q = sample_pair(Xs_ctgan, Xs_test)
    dcr_c_s2r, scaler_c = dcr_to_reference(Xc_q, Xs_train, scaler=None)
    dcr_c_t2r, _ = dcr_to_reference(Xt_q, Xs_train, scaler=scaler_c)
    dcr_c_r2r = r2r_baseline(Xs_train, scaler_c)

    mpl.rcParams["font.family"] = "Liberation Sans"
    mpl.rcParams["font.size"] = 8.5
    mpl.rcParams["axes.titlesize"] = 8.5
    mpl.rcParams["axes.labelsize"] = 8.5
    mpl.rcParams["xtick.labelsize"] = 8.5
    mpl.rcParams["ytick.labelsize"] = 8.5
    mpl.rcParams["legend.fontsize"] = 8.5
    mpl.rcParams["text.usetex"] = False

    fig_w = 16.0 / 2.54
    fig_h = 3.0 / 2.54
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), sharey=True)
    plt.subplots_adjust(wspace=0.35, right=0.80)

    panels = [
        ("GaussDiff", dcr_g_s2r, dcr_g_t2r, dcr_g_r2r),
        ("TVAE", dcr_v_s2r, dcr_v_t2r, dcr_v_r2r),
        ("CTGAN", dcr_c_s2r, dcr_c_t2r, dcr_c_r2r),
    ]

    for ax, (name, s2r, t2r, r2r) in zip(axes, panels):
        x, y = ecdf(s2r)
        ax.plot(x, y, linestyle="-", linewidth=1, label="Synthetic → Train (S2R)")
        x, y = ecdf(t2r)
        ax.plot(x, y, linestyle="--", linewidth=1, label="Test → Train (T2R)")
        x, y = ecdf(r2r)
        ax.plot(x, y, linestyle=":", linewidth=1, label="Train → Train (R2R)")
        ax.set_title(name, pad=2)
        ax.set_xlabel("DCR")
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 1.01)

    axes[0].set_ylabel("ECDF")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="center left", bbox_to_anchor=(0.82, 0.5))

    out_png = PLOT_DCR_DIR / f"dcr_sample_{sample_id_:02d}.png"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

print("DCR done")

PLOT_DIST_DIR = ensure_dir(Path("plot/dist"))

df_origin = pd.concat(
    [
        pd.read_csv("data/2_split/X_num_train.csv", index_col=0),
        pd.read_csv("data/2_split/X_cat_train.csv", index_col=0),
    ],
    axis=1,
)

NUMERIC_COLS = [
    "Age (year)", "Height (cm)", "Weight (cm)", "Waist (cm)",
    "SBP (mmHg)", "DBP (mmHg)", "FBG (mg/dL)", "TG (mg/dL)",
    "TC (mg/dL)", "HDL-C (mg/dL)", "LDL-C (mg/dL)", "CRP (mg/L)",
]
CAT_COLS = ["Gender", "DM", "HTN", "DYL"]

LABEL_MAPS = {
    "Gender": {0: "Female", 1: "Male"},
    "DM": {0: "No", 1: "Yes"},
    "HTN": {0: "No", 1: "Yes"},
    "DYL": {0: "No", 1: "Yes"},
}

CM_TO_INCH = 1.0 / 2.54

def _cm2inch(x_cm: float) -> float:
    return x_cm * CM_TO_INCH

def _kde_line_and_fill(ax, x, label, lw=1.2, alpha=0.25, grid_size=256):
    x = pd.Series(x).dropna().astype(float).to_numpy()
    x = x[np.isfinite(x)]
    if x.size < 3:
        return None

    xmin = np.min(x)
    xmax = np.max(x)
    if xmin == xmax:
        xmin -= 1e-6
        xmax += 1e-6

    pad = 0.05 * (xmax - xmin)
    lo = xmin - pad
    hi = xmax + pad
    grid = np.linspace(lo, hi, grid_size)

    kde = gaussian_kde(x)
    dens = kde(grid)

    (line,) = ax.plot(grid, dens, lw=lw, label=label)
    ax.fill_between(grid, 0, dens, color=line.get_color(), alpha=alpha, linewidth=0)
    return line

def plot_numeric_kde_grid(
    df_origin_: pd.DataFrame,
    df_synth_: pd.DataFrame,
    numeric_cols,
    out_png: Path | None = None,
    width_cm: float = 16.0,
    font_size: int = 8,
    ncols: int = 4,
    dpi: int = 600,
    x_limits: dict | None = None,
    fill_alpha: float = 0.25,
):
    mpl.rcParams["font.family"] = "Liberation Sans"
    mpl.rcParams["font.size"] = font_size
    sns.set_style("white")

    nvars = len(numeric_cols)
    nrows = int(np.ceil(nvars / float(ncols)))
    height_cm = max(6.0, 3.0 * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(_cm2inch(width_cm), _cm2inch(height_cm)),
    )
    axes = np.array(axes).reshape(nrows, ncols)

    legend_handles = None
    legend_labels = None

    for i, col in enumerate(numeric_cols):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        h1 = _kde_line_and_fill(ax, df_origin_[col], "Original", alpha=fill_alpha)
        h2 = _kde_line_and_fill(ax, df_synth_[col], "Synthetic", alpha=fill_alpha)

        if isinstance(x_limits, dict) and (col in x_limits) and (x_limits[col] is not None):
            ax.set_xlim(x_limits[col][0], x_limits[col][1])

        ax.set_title(f"{col}", pad=3.0)

        if c == 0:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("")
            ax.set_yticks([])

        if r != nrows - 1:
            ax.set_xticks([])

        if (legend_handles is None) and (h1 is not None) and (h2 is not None):
            legend_handles = [h1, h2]
            legend_labels = ["Original", "Synthetic"]

        sns.despine(ax=ax)

    for j in range(nvars, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
        )

    fig.tight_layout(pad=0.6, w_pad=0.6, h_pad=0.8)

    if out_png is not None:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    return fig

def plot_categorical_bar_grid(
    df_origin_: pd.DataFrame,
    df_synth_: pd.DataFrame,
    cat_cols,
    out_png: Path | None = None,
    width_cm: float = 16.0,
    font_size: int = 8,
    ncols: int = 4,
    dpi: int = 600,
    label_maps: dict | None = None,
):
    mpl.rcParams["font.family"] = "Liberation Sans"
    mpl.rcParams["font.size"] = font_size
    sns.set_style("white")

    nvars = len(cat_cols)
    nrows = int(np.ceil(nvars / float(ncols)))
    height_cm = max(5.0, 3.0 * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(_cm2inch(width_cm), _cm2inch(height_cm)),
    )
    axes = np.array(axes).reshape(nrows, ncols)

    handles_for_legend = None
    labels_for_legend = None

    for i, col in enumerate(cat_cols):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        s1 = df_origin_[col].dropna()
        s2 = df_synth_[col].dropna()

        if isinstance(label_maps, dict) and (col in label_maps) and isinstance(label_maps[col], dict):
            s1 = s1.map(label_maps[col]).fillna(s1)
            s2 = s2.map(label_maps[col]).fillna(s2)

        cats = list(pd.Index(pd.concat([s1, s2], axis=0).unique()))
        pref = ["No", "Yes", "Female", "Male", 0, 1]
        cats_sorted = []
        for p in pref:
            if p in cats and p not in cats_sorted:
                cats_sorted.append(p)
        for x in cats:
            if x not in cats_sorted:
                cats_sorted.append(x)

        c1 = s1.value_counts().reindex(cats_sorted).fillna(0).astype(int)
        c2 = s2.value_counts().reindex(cats_sorted).fillna(0).astype(int)

        plot_df = pd.DataFrame(
            {
                "Category": np.concatenate([c1.index.values, c2.index.values]),
                "Count": np.concatenate([c1.values, c2.values]),
                "Dataset": ["Original"] * len(c1) + ["Synthetic"] * len(c2),
            }
        )

        sns.barplot(data=plot_df, x="Category", y="Count", hue="Dataset", ax=ax)

        ax.set_title(f"{col}", pad=6.0)

        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2.0,
                    h,
                    f"{int(h)}",
                    ha="center",
                    va="bottom",
                    fontsize=font_size,
                )

        if c == 0:
            ax.set_ylabel("Count")
        else:
            ax.set_ylabel("")
            ax.set_yticks([])

        if r != nrows - 1:
            ax.set_xticklabels([])
            ax.set_xticks([])

        if ax.get_legend() is not None:
            if handles_for_legend is None:
                handles_for_legend, labels_for_legend = ax.get_legend_handles_labels()
            ax.get_legend().remove()

        sns.despine(ax=ax)

    for j in range(nvars, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    if handles_for_legend and labels_for_legend:
        fig.legend(
            handles_for_legend,
            labels_for_legend,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
        )

    fig.tight_layout(pad=0.6, w_pad=0.6, h_pad=0.8)

    if out_png is not None:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    return fig

N_DIST = 15
for sid in tqdm(range(N_DIST), desc="Dist"):
    xrev = Path(f"results/sample/sample_{sid:02d}/X_reversed.csv")
    require_file(xrev, "Distribution plots require results/sample/sample_XX/X_reversed.csv.")
    df_gauss = pd.read_csv(xrev)

    fig1 = plot_numeric_kde_grid(
        df_origin,
        df_gauss,
        NUMERIC_COLS,
        out_png=PLOT_DIST_DIR / f"num_{sid:02d}.png",
        width_cm=16.0,
        font_size=8,
        ncols=4,
        dpi=600,
        x_limits=None,
        fill_alpha=0.25,
    )
    plt.close(fig1)

    fig2 = plot_categorical_bar_grid(
        df_origin,
        df_gauss,
        CAT_COLS,
        out_png=PLOT_DIST_DIR / f"cat_{sid:02d}.png",
        width_cm=16.0,
        font_size=8,
        ncols=4,
        dpi=600,
        label_maps=LABEL_MAPS,
    )
    plt.close(fig2)

print("Done.")
