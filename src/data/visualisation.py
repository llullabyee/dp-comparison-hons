from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import pyarrow as pa
import pyarrow.csv as pv

from src.utils.config import MECHANISMS, RESULT_DIR
from src.utils.utility_helpers import (
    METRIC_DISPLAY_NAMES,
    METRIC_KEYS,
    SCALE_DEPENDENT_METRICS,
)

eps = [1.0, 2.0, 5.0, 10.0]
window_size = 3
peak = 1500
sensitivity = [1500, 2000, 3000]

POWER_LABELS = ["dplm", "gai_power"]
COUNTS_LABELS_MAINS_ONLY = ["gai_counts"] + [f"{m}_mains" for m in MECHANISMS]
COUNTS_LABELS_APPLIANCES_ONLY = [f"{m}_appliances" for m in MECHANISMS]
COUNTS_LABELS_COMBINED = (
    COUNTS_LABELS_MAINS_ONLY + COUNTS_LABELS_APPLIANCES_ONLY
)

DAY_VIEW_LABELS = ["dplm", "gai_power"]
DAY_VIEW_EPS = None  # e.g. {1.0, 5.0, 10.0} to restrict, or None for all

BRANCH_DISPLAY_NAMES = {
    "power": "Power",
    "counts_mains": "Counts (Mains)",
    "counts_appliances": "Counts (Appliances)",
    "counts_combined": "Counts (Mains + Appliances)",
    "all": "",
}

# Dark24 gives 24 visually distinct colors, comfortably covers any label list here
DISTINCT_COLORS = pc.qualitative.Dark24

FIG_WIDTH = 1000
FIG_HEIGHT = 700
FIG_SCALE = 1.5  # PDF is vector; keep modest, this just scales page size
FONT_FAMILY = "CMU Serif"

html_dir = RESULT_DIR / "figures" / "html"
pdf_dir = RESULT_DIR / "figures" / "pdf"
tables_dir = RESULT_DIR / "figures" / "tables"
html_dir.mkdir(parents=True, exist_ok=True)
pdf_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)


def display_title(metric: str, title_suffix: str) -> str:
    m = METRIC_DISPLAY_NAMES.get(metric, metric.replace("_", " ").title())
    b = BRANCH_DISPLAY_NAMES.get(title_suffix, title_suffix.title())
    return f"{m} — {b}".rstrip(" —") if b else m


def display_label(label: str) -> str:
    fixed = {
        "gai_counts": "Gai (Counts)",
        "gai_power": "Gai (Power)",
        "dplm": "DPLM",
    }
    if label in fixed:
        return fixed[label]
    if "_" in label:
        mech, bin_type = label.split("_", 1)
        return f"{mech} ({bin_type.title()})"
    return label


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def agg_table_stats(table: pa.Table) -> pd.DataFrame:
    """One table (rows=days) -> mean/std per column (excluding 'day')."""
    df = table.to_pandas().drop(columns="day")
    return pd.DataFrame({"mean": df.mean(), "std": df.std()})


def _apply_layout(
    fig,
    title,
    xaxis_title,
    yaxis_title,
    xaxis_type=None,
    tickangle=None,
    corner_text=None,
):
    annotations = []
    if corner_text:
        annotations.append(
            dict(
                text=corner_text,
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.12,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=15, family=FONT_FAMILY, color="rgba(0,0,0,0.6)"),
            )
        )
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, family=FONT_FAMILY)),
        annotations=annotations,
        xaxis=dict(
            title=xaxis_title,
            type=xaxis_type,
            tickangle=tickangle,
            showline=True,
            linecolor="rgba(0,0,0,0.3)",
            mirror=True,
            ticks="outside",
        ),
        yaxis=dict(
            title=yaxis_title,
            showline=True,
            linecolor="rgba(0,0,0,0.3)",
            mirror=True,
            ticks="outside",
        ),
        template="plotly_white",
        font=dict(family=FONT_FAMILY, size=15, color="black"),
        legend=dict(font=dict(size=13, family=FONT_FAMILY)),
        margin=dict(l=70, r=30, t=90, b=70),
        width=FIG_WIDTH,
        height=FIG_HEIGHT,
    )
    return fig


def plot_vs_epsilon(
    eps_sorted, tables_sorted, labels, title, metric, corner_text=None
):
    """One point (mean ± std across days) per epsilon, per label."""
    stats_per_eps = [agg_table_stats(t) for t in tables_sorted]
    fig = go.Figure()
    for i, label in enumerate(labels):
        if not all(label in s.index for s in stats_per_eps):
            continue
        means = np.array([s.loc[label, "mean"] for s in stats_per_eps])
        stds = np.array([s.loc[label, "std"] for s in stats_per_eps])
        upper = means + stds
        lower = means - stds
        color = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]

        fig.add_trace(
            go.Scatter(
                x=eps_sorted,
                y=means,
                mode="lines+markers",
                name=display_label(label),
                marker=dict(size=8, color=color),
                line=dict(width=2.5, color=color),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=eps_sorted + eps_sorted[::-1],
                y=list(upper) + list(lower[::-1]),
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.15),
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    yaxis_label = METRIC_DISPLAY_NAMES.get(metric, metric)
    return _apply_layout(
        fig,
        title,
        "Privacy Budget (ε)",
        yaxis_label,
        xaxis_type="log",
        corner_text=corner_text,
    )


def plot_vs_day(
    eps_sorted,
    tables_sorted,
    label,
    title,
    metric,
    eps_subset=None,
    corner_text=None,
):
    """One line per epsilon, day on x-axis, for a single label."""
    fig = go.Figure()
    i = 0
    for e, table in zip(eps_sorted, tables_sorted):
        if eps_subset is not None and e not in eps_subset:
            continue
        df = table.to_pandas().set_index("day")
        if label not in df.columns:
            continue
        color = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[label],
                mode="lines+markers",
                name=f"ε={e:g}",
                marker=dict(size=6, color=color),
                line=dict(width=2, color=color),
            )
        )
        i += 1
    yaxis_label = METRIC_DISPLAY_NAMES.get(metric, metric)
    return _apply_layout(
        fig,
        title,
        "Day",
        yaxis_label,
        tickangle=-45,
        corner_text=corner_text,
    )


def heatmap_metric_vs_eps(
    eps_sorted, tables_sorted, labels, title, metric, corner_text=None
):
    """rows = labels, cols = epsilon, color = mean across days (linear scale)."""
    z = []
    for label in labels:
        row = [
            table.to_pandas()[label].mean()
            if label in table.column_names
            else np.nan
            for table in tables_sorted
        ]
        z.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[f"{e:g}" for e in eps_sorted],
            y=[display_label(l) for l in labels],
            colorscale="RdBu_r" if metric == "bias" else "Viridis",
            zmid=0 if metric == "bias" else None,
            colorbar=dict(title=METRIC_DISPLAY_NAMES.get(metric, metric)),
            text=[
                [f"{v:,.0f}" if not np.isnan(v) else "" for v in row]
                for row in z
            ],
            texttemplate="%{text}",
            textfont=dict(size=11),
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _apply_layout(
        fig,
        title,
        "Privacy Budget (ε)",
        "Label",
        corner_text=corner_text,
    )


def heatmap_metric_vs_eps_combined(
    eps_sorted, tables_sorted, labels, title, metric, corner_text=None
):
    """rows = labels (mains + appliances mixed scale), cols = epsilon, log-color
    to make discrepancy between branches visible without one drowning the other."""
    z = []
    for label in labels:
        row = [
            table.to_pandas()[label].mean()
            if label in table.column_names
            else np.nan
            for table in tables_sorted
        ]
        z.append(row)
    z = np.array(z, dtype=float)

    if metric == "bias":
        z_display = z
        colorscale, zmid = "RdBu_r", 0
        colorbar_title = METRIC_DISPLAY_NAMES.get(metric, metric)
    else:
        z_display = np.log10(np.clip(z, 1e-9, None))
        colorscale, zmid = "Viridis", None
        colorbar_title = f"log₁₀({METRIC_DISPLAY_NAMES.get(metric, metric)})"

    fig = go.Figure(
        data=go.Heatmap(
            z=z_display,
            x=[f"{e:g}" for e in eps_sorted],
            y=[display_label(l) for l in labels],
            colorscale=colorscale,
            zmid=zmid,
            colorbar=dict(title=colorbar_title),
            text=[
                [f"{v:,.0f}" if not np.isnan(v) else "" for v in row]
                for row in z
            ],
            texttemplate="%{text}",
            textfont=dict(size=10),
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _apply_layout(
        fig,
        title,
        "Privacy Budget (ε)",
        "Label",
        corner_text=corner_text,
    )


def save_fig(fig, name):
    fig.write_html(html_dir / f"{name}.html")
    fig.write_image(pdf_dir / f"{name}.pdf", scale=FIG_SCALE)  # needs kaleido


def make_figures(
    metric,
    eps_sorted,
    tables_sorted,
    labels,
    title_suffix,
    s,
    day_view_labels=None,
    day_view_eps=None,
):
    """Emit the epsilon-sweep line figure, heatmap, and optional day-view diagnostic."""
    title = display_title(metric, title_suffix)
    corner_text = f"λ={s:g}"

    fig = plot_vs_epsilon(
        eps_sorted,
        tables_sorted,
        labels,
        title,
        metric,
        corner_text=corner_text,
    )
    save_fig(fig, f"{metric}_{title_suffix}_sens={s:g}")

    fig = heatmap_metric_vs_eps(
        eps_sorted,
        tables_sorted,
        labels,
        f"{title} — Heatmap",
        metric,
        corner_text=corner_text,
    )
    save_fig(fig, f"{metric}_{title_suffix}_heatmap_sens={s:g}")

    for label in day_view_labels or []:
        if label not in labels:
            continue
        fig = plot_vs_day(
            eps_sorted,
            tables_sorted,
            label,
            f"{title} — {display_label(label)} by Day",
            metric,
            eps_subset=day_view_eps,
            corner_text=corner_text,
        )
        save_fig(fig, f"{metric}_{title_suffix}_{label}_byday_sens={s:g}")


# summary[(s, e)][label][metric] = mean across days
summary = defaultdict(lambda: defaultdict(dict))

for metric in METRIC_KEYS:
    for s in sensitivity:
        fps = [
            RESULT_DIR
            / f"eps={e:g}"
            / f"w={window_size:g}"
            / f"peak={peak:g}"
            / f"sens={s:g}"
            / "utility"
            / f"{metric}.csv"
            for e in eps
        ]
        tables = [pv.read_csv(fp) for fp in fps]

        order = np.argsort(eps)
        eps_sorted = [eps[i] for i in order]
        tables_sorted = [tables[i] for i in order]

        # accumulate per-label mean for the summary tables, regardless of branch
        for e, table in zip(eps_sorted, tables_sorted):
            df = table.to_pandas().drop(columns="day")
            for label, val in df.mean().items():
                summary[(s, e)][label][metric] = val

        if metric in SCALE_DEPENDENT_METRICS:
            all_labels = set(tables_sorted[0].column_names) - {"day"}
            power_cols = [c for c in POWER_LABELS if c in all_labels]
            counts_cols_mains = [
                c for c in COUNTS_LABELS_MAINS_ONLY if c in all_labels
            ]
            counts_cols_appliances = [
                c for c in COUNTS_LABELS_APPLIANCES_ONLY if c in all_labels
            ]
            counts_cols_combined = [
                c for c in COUNTS_LABELS_COMBINED if c in all_labels
            ]

            if power_cols:
                make_figures(
                    metric,
                    eps_sorted,
                    tables_sorted,
                    power_cols,
                    "power",
                    s,
                    day_view_labels=DAY_VIEW_LABELS,
                    day_view_eps=DAY_VIEW_EPS,
                )
            if counts_cols_mains:
                make_figures(
                    metric,
                    eps_sorted,
                    tables_sorted,
                    counts_cols_mains,
                    "counts_mains",
                    s,
                    day_view_labels=DAY_VIEW_LABELS,
                    day_view_eps=DAY_VIEW_EPS,
                )
            if counts_cols_appliances:
                make_figures(
                    metric,
                    eps_sorted,
                    tables_sorted,
                    counts_cols_appliances,
                    "counts_appliances",
                    s,
                    day_view_labels=DAY_VIEW_LABELS,
                    day_view_eps=DAY_VIEW_EPS,
                )
            if counts_cols_combined:
                title = display_title(metric, "counts_combined")
                fig = heatmap_metric_vs_eps_combined(
                    eps_sorted,
                    tables_sorted,
                    counts_cols_combined,
                    f"{title} — Heatmap",
                    metric,
                    corner_text=f"λ={s:g}",
                )
                save_fig(fig, f"{metric}_counts_combined_heatmap_sens={s:g}")
        else:
            all_labels = [
                c for c in tables_sorted[0].column_names if c != "day"
            ]
            make_figures(
                metric,
                eps_sorted,
                tables_sorted,
                all_labels,
                "all",
                s,
                day_view_labels=DAY_VIEW_LABELS,
                day_view_eps=DAY_VIEW_EPS,
            )

# emit one summary CSV per (sensitivity, epsilon): rows = label, columns = metric
for (s, e), label_dict in summary.items():
    df = pd.DataFrame(
        label_dict
    ).T  # dict-of-dicts -> outer keys as columns, so .T flips it
    df.index = [display_label(l) for l in df.index]
    df.index.name = "label"
    df.columns = [METRIC_DISPLAY_NAMES.get(c, c) for c in df.columns]
    df.to_csv(tables_dir / f"summary_sens={s:g}_eps={e:g}.csv")
