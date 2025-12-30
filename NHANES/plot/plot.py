

wd = u"/media/jie/expand_5t/7exp_expand/tang_all/tang_v2/GaussDiff/NHANES"
sd = 42
n_row = 6
n_col = 20

from IPython.display import display
import os, sys, warnings
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

if not sys.warnoptions:
    warnings.simplefilter("ignore")

np.random.seed(sd)
pd.set_option("display.max_columns", n_col)
pd.set_option("display.min_rows", n_row)

print("current:", os.getcwd())
os.chdir(wd)
print("defined:", wd)

def my_makedir(customized_path, f=False):
    if (os.path.exists(customized_path)) and (f == False):
        print("\nFolder '{}' exists.".format(customized_path))
    else:
        os.makedirs(customized_path)
        print("\nFolder '{}' created.".format(customized_path))
    print("\n" + "-" * 60)

def my_create_file(file_path):
    if os.path.isfile(file_path):
        os.system("rm -rf %s" % file_path)
    print("remove_create: %s" % file_path)

def cut_value_counts(data, bins, verbose=False):
    data_binned = pd.cut(data, bins=bins)
    counts = data_binned.value_counts().reset_index()
    counts.columns = ["bin", "count"]
    counts = counts.sort_values(by="bin")
    if verbose:
        print(counts)
    return counts

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from scipy import stats
from scipy.stats import ttest_ind

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.lines import Line2D

from scipy.stats import gaussian_kde

df_met = pd.read_csv("results/eval/metrics_all.csv")

TYPE_ALIAS = {
    "Random_Oversampling": "Random oversampling",
    "Random oversampling": "Random oversampling",
}
if "type" in df_met.columns:
    df_met["type"] = df_met["type"].replace(TYPE_ALIAS)

os.makedirs("plot/metrics/", exist_ok=True)

type_col = "type"

row_specs = [
    ("GaussDiff", "GaussDiff", "global"),
    ("Baseline",  "TVAE", "TVAE"),
    ("Baseline",  "CTGAN", "CTGAN"),
    ("Baseline",  "SVMSMOTE", "SVMSMOTE"),
    ("Baseline",  "SMOTE", "SMOTE"),
    ("Baseline",  "ADASYN", "ADASYN"),
    ("Baseline",  "Random oversampling", "Random oversampling"),
]

metric_blocks = [
    ("recall", "Recall"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
]

B = 10000
alpha = 0.05
seed = 2025
rng = np.random.default_rng(seed)

def bootstrap_ci_mean(x: np.ndarray, rng, B: int = 10000, alpha: float = 0.05):
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    mu = float(np.mean(x))
    sd_ = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n == 1:
        return mu, sd_, np.nan, np.nan, n
    idx = rng.integers(0, n, size=(B, n))
    boot_means = x[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return mu, sd_, lo, hi, n

def fmt_mean_sd(mu, sd_, nd=3):
    if not np.isfinite(mu):
        return "NA"
    if not np.isfinite(sd_):
        return f"{mu:.{nd}f}"
    return f"{mu:.{nd}f} ± {sd_:.{nd}f}"

def fmt_ci(lo, hi, nd=3):
    if np.isfinite(lo) and np.isfinite(hi):
        return f"[{lo:.{nd}f}, {hi:.{nd}f}]"
    return "NA"

present_types = set(df_met[type_col].dropna().unique())

boot_stats = {}
for t in present_types:
    g = df_met.loc[df_met[type_col] == t]
    for m, _ in metric_blocks:
        if m not in g.columns:
            raise ValueError(f"Missing column in metrics_all.csv: {m}")
        x = g[m].to_numpy(dtype=float)
        mu, sd_, lo, hi, n = bootstrap_ci_mean(x, rng=rng, B=B, alpha=alpha)
        boot_stats[(t, m)] = (mu, sd_, lo, hi, n)

rows = []
for group, t, display_name in row_specs:
    if t not in present_types:
        continue
    row = {"Group": group, "": display_name}
    for m, mdisp in metric_blocks:
        mu, sd_, lo, hi, n = boot_stats[(t, m)]
        row[(mdisp, "mean ± sd")] = fmt_mean_sd(mu, sd_, nd=3)
        row[(mdisp, "95% confidence interval")] = fmt_ci(lo, hi, nd=3)
    rows.append(row)

df_out = pd.DataFrame(rows)
left_cols = ["Group", ""]
metric_cols = [c for c in df_out.columns if isinstance(c, tuple)]
df_out = df_out[left_cols + metric_cols]
df_out.loc[df_out["Group"].duplicated(), "Group"] = ""

out_csv = "plot/metrics/metrics_summary_bootstrap_ci95_wide.csv"
df_out.to_csv(out_csv, index=False)
print(df_out.to_string(index=False))
print("\nsaved:", out_csv)
print(f"\nBootstrap settings: B={B}, alpha={alpha}, seed={seed}")

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.size"] = 8.5
mpl.rcParams["axes.titlesize"] = 8.0
mpl.rcParams["axes.labelsize"] = 8.5
mpl.rcParams["xtick.labelsize"] = 8.0
mpl.rcParams["ytick.labelsize"] = 7.0
mpl.rcParams["text.usetex"] = False
sns.set(style="white")

order = [
    "GaussDiff",
    "GaussDiff (unaligned)",
    "GaussDiff (local)",
    "TVAE",
    "CTGAN",
    "ADASYN",
    "SMOTE",
    "Random oversampling",
]
present = set(df_met[type_col].dropna().unique())
order_use = [t for t in order if t in present]

warm = sns.color_palette("Reds", 6)[2:5]
cool = sns.color_palette("Set2", 10)

palette_map = {
    "GaussDiff": warm[2],
    "GaussDiff (unaligned)": warm[1],
    "GaussDiff (local)": warm[0],
    "TVAE": cool[2],
    "CTGAN": cool[4],
    "ADASYN": cool[6],
    "SMOTE": cool[7],
    "Random oversampling": cool[8],
}
for t in order_use:
    palette_map.setdefault(t, cool[0])
palette_list = [palette_map[t] for t in order_use]

metrics_list = ["pr_auc", "roc_auc", "recall"]
y_labels = ["PR-AUC(→)", "ROC-AUC(→)", "Recall(→)"]

df_melt = df_met.melt(
    id_vars=[type_col],
    value_vars=metrics_list,
    var_name="metrics",
    value_name="value",
).rename(columns={type_col: "model"})

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

for i, (ax, metric) in enumerate(zip(axes, metrics_list)):
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

    ax.set_ylabel(y_labels[i], fontsize=8.0)
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
    ax.spines["right"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    y0, y1 = ax.get_ylim()
    span = y1 - y0
    ax.set_ylim(y0, y1 + 0.22 * span)
    y0, y1 = ax.get_ylim()
    span = y1 - y0

    y_base = y0 + 0.03 * span
    y_up = 0.2 * span

    for j, model in enumerate(order_use):
        mu = metric_summary.loc[model, "mean"]
        sdv = metric_summary.loc[model, "std"]
        if not np.isfinite(mu):
            continue
        txt = f"{mu:.3f}±{sdv:.3f}"
        move_up = (metric == "recall" and mu < 0.5) or (metric == "pr_auc" and mu < 0.15)
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

out_png = "plot/metrics/metrics.png"
fig.savefig(out_png, dpi=600, bbox_inches="tight")
plt.close(fig)
print("\nsaved:", out_png)

y_true = pd.read_csv("data/2_split/y_test.csv", index_col=0)
df_duration = pd.read_csv("data/4_km/km_event_year.csv", index_col=0)

n = len(np.sort(glob("results/eval/sample_*/")))

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

os.makedirs("plot/km", exist_ok=True)

for sample_id_ in tqdm(range(n)):
    dfs = [y_true, df_duration]
    for disp_name, folder in PROBA_SPECS:
        p_path = f"results/eval/sample_{sample_id_:02d}/{folder}/y_proba.csv"
        if not os.path.exists(p_path):
            raise FileNotFoundError(p_path)
        d = pd.read_csv(p_path)
        d.columns = [disp_name]
        d.index = y_true.index
        dfs.append(d)

    df_km = pd.concat(dfs, axis=1)
    df_km2 = df_km.loc[df_km.notnull().all(axis=1), :].copy()

    assert ((df_km2["CVD"] != df_km2["events"]).sum() == 0)
    df_km2.drop("CVD", axis=1, inplace=True)

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

    fig.savefig(f"plot/km/km_sample_{sample_id_:02d}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

print("KM done")

import joblib
from scipy.special import inv_boxcox
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

os.makedirs("plot/dcr", exist_ok=True)

mapping = {"A": 0, "B": 1, "No": 0, "Yes": 1, "Male": 0, "Female": 1}

n = len(np.sort(glob("results/eval/sample_*/")))

for sample_id_ in tqdm(range(n)):

    real_data_dir = "data/2_split"
    out_png = f"plot/dcr/dcr_sample_{sample_id_:02d}.png"
    label_column = "CVD"

    def create_real_dataset(split_set):
        num_path = f"{real_data_dir}/X_num_{split_set}.csv"
        cat_path = f"{real_data_dir}/X_cat_{split_set}.csv"
        y_path = f"{real_data_dir}/y_{split_set}.csv"

        X_num = pd.read_csv(num_path, index_col=0)
        X_cat = pd.read_csv(cat_path, index_col=0)
        y = pd.read_csv(y_path, index_col=0)

        X_all = pd.concat([X_num, X_cat], axis=1)
        return X_all, y[label_column]

    Xs_train, y_real_train = create_real_dataset("train")
    Xs_test, y_real_test = create_real_dataset("test")

    generated_dir_gauss = f"results/sample/sample_{sample_id_:02d}"
    tvae_dir = f"results/sample_tvae/sample_{sample_id_:02d}"
    ctgan_dir = f"results/sample_ctgan/sample_{sample_id_:02d}"

    def read_generated_gauss(generated_dir):
        X_gen = pd.read_csv(f"{generated_dir}/X_reversed.csv")
        y_gen = pd.read_csv(f"{generated_dir}/y_.csv")[label_column]
        X_gen = X_gen.reindex(columns=Xs_test.columns)
        obj_cols = X_gen.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            X_gen[c] = X_gen[c].map(mapping)
        return X_gen, y_gen

    def read_generated_baseline(basedir):
        X = pd.read_csv(f"{basedir}/X_.csv")
        y = pd.read_csv(f"{basedir}/y_.csv")[label_column]
        X = X.reindex(columns=Xs_test.columns)
        obj_cols = X.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            X[c] = X[c].map(mapping)
        return X, y

    Xs_gauss, y_gauss = read_generated_gauss(generated_dir_gauss)
    Xs_tvae, y_tvae = read_generated_baseline(tvae_dir)
    Xs_ctgan, y_ctgan = read_generated_baseline(ctgan_dir)

    def _clean_finite(X):
        X = X.copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        return X

    def dcr_to_reference(X_query, X_ref, scaler=None):
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

    def r2r_baseline(X_train, scaler):
        X_train_clean = _clean_finite(X_train).dropna(axis=0, how="any")
        X_train_z = scaler.transform(X_train_clean)
        tree = cKDTree(X_train_z)
        d2, _ = tree.query(X_train_z, k=2)
        return d2[:, 1]

    rng = np.random.default_rng(42)

    def sample_pair(X_syn, X_test):
        n_ = min(len(X_syn), len(X_test))
        idx_syn = rng.choice(len(X_syn), size=n_, replace=False)
        idx_test = rng.choice(len(X_test), size=n_, replace=False)
        return X_syn.iloc[idx_syn], X_test.iloc[idx_test]

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

    def ecdf(x):
        x = np.sort(np.asarray(x))
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

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
        x, y = ecdf(s2r); ax.plot(x, y, linestyle="-", linewidth=1, label="Synthetic → Train (S2R)")
        x, y = ecdf(t2r); ax.plot(x, y, linestyle="--", linewidth=1, label="Test → Train (T2R)")
        x, y = ecdf(r2r); ax.plot(x, y, linestyle=":", linewidth=1, label="Train → Train (R2R)")
        ax.set_title(name, pad=2)
        ax.set_xlabel("DCR")
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 1.01)

    axes[0].set_ylabel("ECDF")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="center left", bbox_to_anchor=(0.82, 0.5))
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

print("DCR done")

df_origin = pd.concat(
    [
        pd.read_csv("data/2_split/X_num_train.csv", index_col=0),
        pd.read_csv("data/2_split/X_cat_train.csv", index_col=0),
    ],
    axis=1,
)

df_origin = df_origin.rename(columns={"BG2h": "BG2h (mg/dL)"})

for sd_i_ in tqdm(range(15)):

    df_gaussdiff = pd.read_csv(f"results/sample/sample_{sd_i_:02d}/X_reversed.csv")
    df_gaussdiff = df_gaussdiff.rename(columns={"BG2h": "BG2h (mg/dL)"})

    mpl.rcParams["font.family"] = "Liberation Sans"
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["axes.titlesize"] = 8
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8
    mpl.rcParams["legend.fontsize"] = 8
    mpl.rcParams["text.usetex"] = False
    sns.set_style("white")

    CM_TO_INCH = 1.0 / 2.54

    def _cm2inch(x_cm):
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
        df_origin,
        df_synth,
        numeric_cols,
        out_png=None,
        width_cm=16.0,
        font_size=8,
        ncols=4,
        dpi=600,
        x_limits=None,
        fill_alpha=0.25,
    ):
        mpl.rcParams["font.size"] = font_size

        nvars = len(numeric_cols)
        nrows = int(np.ceil(nvars / float(ncols)))

        height_cm = max(6.0, 3.0 * nrows)
        fig_w = _cm2inch(width_cm)
        fig_h = _cm2inch(height_cm)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
        axes = np.array(axes).reshape(nrows, ncols)

        legend_handles = None
        legend_labels = None

        for i, col in enumerate(numeric_cols):
            r = i // ncols
            c = i % ncols
            ax = axes[r, c]

            x1 = df_origin[col]
            x2 = df_synth[col]

            h1 = _kde_line_and_fill(ax, x1, label="Original", lw=1.2, alpha=fill_alpha)
            h2 = _kde_line_and_fill(ax, x2, label="Synthetic", lw=1.2, alpha=fill_alpha)

            if isinstance(x_limits, dict) and (col in x_limits) and (x_limits[col] is not None):
                ax.set_xlim(x_limits[col][0], x_limits[col][1])

            ax.set_title(f"{col}", pad=3.0)

            if c == 0:
                ax.set_ylabel("Density")
            else:
                ax.set_ylabel("")
                ax.set_yticks([])

            if r == nrows - 1:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("")
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

        if out_png:
            fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_categorical_bar_grid(
        df_origin,
        df_synth,
        cat_cols,
        out_png=None,
        width_cm=16.0,
        font_size=8,
        ncols=4,
        dpi=600,
        label_maps=None,
    ):
        mpl.rcParams["font.size"] = font_size

        nvars = len(cat_cols)
        nrows = int(np.ceil(nvars / float(ncols)))

        height_cm = max(5.0, 3.0 * nrows)
        fig_w = _cm2inch(width_cm)
        fig_h = _cm2inch(height_cm)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
        axes = np.array(axes).reshape(nrows, ncols)

        handles_for_legend = None
        labels_for_legend = None

        for i, col in enumerate(cat_cols):
            r = i // ncols
            c = i % ncols
            ax = axes[r, c]

            s1 = df_origin[col].dropna()
            s2 = df_synth[col].dropna()

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

            ax.set_title(f"{col}", pad=8.0)

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

            if r == nrows - 1:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("")
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

        if out_png:
            fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        return fig

    numeric_cols = df_gaussdiff.columns[:-5]
    cat_cols = df_gaussdiff.columns[-5:]

    label_maps = {
        "Gender": {0: "Female", 1: "Male"},
        "Hypertension": {0: "No", 1: "Yes"},
        "PreDM": {0: "No", 1: "Yes"},
        "DM": {0: "No", 1: "Yes"},
        "DM_PreDM": {0: "No", 1: "Yes"},
    }

    os.makedirs("plot/dist", exist_ok=True)

    fig1 = plot_numeric_kde_grid(
        df_origin,
        df_gaussdiff,
        numeric_cols,
        out_png=os.path.join(".", f"plot/dist/num_{sd_i_:02d}.png"),
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
        df_gaussdiff,
        cat_cols,
        out_png=os.path.join(".", f"plot/dist/cat_{sd_i_:02d}.png"),
        width_cm=16.0,
        font_size=8,
        ncols=5,
        dpi=600,
        label_maps=label_maps,
    )
    plt.close(fig2)

print("Done.")
