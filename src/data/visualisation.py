from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import pyarrow.csv as pv

from src.utils.config import MECHANISMS, RESULT_DIR
from src.utils.utility_helpers import (
    METRIC_DISPLAY_NAMES,
    METRIC_KEYS,
)

# The power totals are energy (Wh), so don't call them "Power Consumption".
METRIC_DISPLAY_NAMES = {
    k: ("Energy Consumption" if v == "Power Consumption" else v)
    for k, v in METRIC_DISPLAY_NAMES.items()
}

eps = [1.0, 2.0, 5.0, 10.0]
window_size = 3
peak = 1500
sensitivity = [1500, 2000, 3000]  # Δ

# output name -> (input subdirectory under .../sens=<s>/, row-index column)
SOURCES = {
    "utility": ("utility", "day"),
    "billing": ("billing_metrics", "home_id"),
}

# Title tag for sources whose figures have no group suffix (billing is monthly).
# The energy-consumption metric is exempt, since "Monthly Power" would be wrong.
SOURCE_TITLE_TAG = {"billing": "Monthly Power"}
ENERGY_METRIC = "power"

# The true-power column is a reference, not a method: it is drawn once as a
# baseline (heatmap row / black crosses) and never as one of the labels.
BASELINE_LABEL = "true_power_consumption"

# Households have no natural order. When True, the by-household scatter sorts
# them by true power, so the estimates scatter around a smooth reference curve.
SORT_HOMES_BY_BASELINE = True

# index column -> (axis title, title suffix, filename suffix)
INDEX_DISPLAY = {
    "day": ("Day", "by Day", "byday"),
    "home_id": ("Home ID", "by Home", "byhome"),
}

# One figure group per comparison. The group name ends up in the title and
# filename, so mains/appliances is conveyed there rather than in the legend.
POWER_LABELS = ["dplm", "gai_power"]
MAINS_LABELS = ["gai_counts"] + [f"{m}_mains" for m in MECHANISMS]
APPLIANCES_LABELS = [f"{m}_appliances" for m in MECHANISMS]

DAY_VIEW_LABELS = ["dplm", "gai_power"]
DAY_VIEW_EPS = None  # e.g. {1.0, 5.0, 10.0} to restrict, or None for all

BRANCH_DISPLAY_NAMES = {
    "power": "Power",
    "mains": "Mains",
    "appliances": "Appliances",
    "all": "",
}

# Legend label for the Gai baselines (both counts and power). Plain HTML italics
# instead of MathJax: MathJax strings don't mix reliably with the trailing
# ", Δ=..." / ", ε=..." text, and this needs no CDN or Kaleido network access.
K_RR = "<i>k</i>-RR"

# Dark24 gives 24 visually distinct colors, comfortably covers any label list here
DISTINCT_COLORS = pc.qualitative.Dark24
# Sensitivity (Δ) / epsilon is encoded by line dash + marker, label by color
DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "x", "cross"]

FIG_WIDTH = 1000
MARGIN_T = 90
MARGIN_B = 70
LEGEND_ROW_PX = 30  # vertical space per legend entry
FIG_HEIGHT = 700
FIG_SCALE = 1.5  # PDF is vector; keep modest, this just scales page size
FONT_FAMILY = "CMU Serif, Latin Modern Roman, Computer Modern, serif"
# Sensitivity symbol used in titles, legends and heatmap rows.
DELTA = "Δ"
SENS_AXIS_TITLE = f"Sensitivity ({DELTA})"
HEATMAP_Y_TITLE = "Method, Sensitivity"
EPS_AXIS_TITLE = "Privacy Budget (ε)"


@dataclass
class Output:
    """figures/<source>/{html,pdf,tables}; html and pdf are grouped by metric
    (and vs_eps/vs_sens sub-folders inside each metric)."""

    root: Path

    @property
    def html(self) -> Path:
        return self.root / "html"

    @property
    def pdf(self) -> Path:
        return self.root / "pdf"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    def make(self):
        for d in (self.html, self.pdf, self.tables):
            d.mkdir(parents=True, exist_ok=True)
        return self


def display_title(metric: str, title_suffix: str, tag=None) -> str:
    m = METRIC_DISPLAY_NAMES.get(metric, metric.replace("_", " ").title())
    b = BRANCH_DISPLAY_NAMES.get(title_suffix, title_suffix.title())
    b = b or tag  # the source tag only fills an otherwise-empty suffix
    return f"{m} — {b}" if b else m


def display_label(label: str) -> str:
    """Legend / axis label for figures. Billing token and the mains/appliances
    suffix are dropped (the figure title carries those). A bare 'billing'
    column becomes '' so the legend shows only Δ."""
    parts = [p for p in label.split("_") if p.lower() != "billing"]
    if not parts:
        return ""
    label = "_".join(parts)

    for suf in ("_mains", "_appliances"):
        if label.endswith(suf):
            label = label[: -len(suf)]
            break

    fixed = {
        "gai_counts": K_RR,
        "gai_power": K_RR,
        "dplm": "DPLM",
        "true_power_consumption": "True Power",
    }
    return fixed.get(label, label)


def table_label(label: str) -> str:
    """Row label for the summary CSVs. Tables mix every group in one file, so
    unlike the figures they must keep power/counts and mains/appliances
    distinct (and stay plain text, no MathJax)."""
    parts = [p for p in label.split("_") if p.lower() != "billing"]
    if parts:
        label = "_".join(parts)
    fixed = {
        "gai_counts": "Gai (Counts)",
        "gai_power": "Gai (Power)",
        "dplm": "DPLM",
        "true_power_consumption": "True Power",
    }
    if label in fixed:
        return fixed[label]
    if "_" in label:
        mech, bin_type = label.split("_", 1)
        return f"{mech} ({bin_type.title()})"
    return label


def _legend(*parts: str) -> str:
    """Join non-empty parts, so a blank label (billing) leaves just 'Δ=1500'."""
    return ", ".join(p for p in parts if p)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _apply_layout(
    fig,
    title,
    xaxis_title,
    yaxis_title,
    tickangle=None,
    height=FIG_HEIGHT,
):
    # Grow the figure so the legend is never clipped: ~LEGEND_ROW_PX per entry
    # plus the top/bottom margins. Bands and heatmaps don't add legend rows.
    n_legend = sum(
        1 for t in fig.data if t.type != "heatmap" and t.showlegend is not False
    )
    height = max(height, n_legend * LEGEND_ROW_PX + MARGIN_T + MARGIN_B + 40)

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, family=FONT_FAMILY)),
        xaxis=dict(
            title=xaxis_title,
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
        margin=dict(l=70, r=30, t=MARGIN_T, b=MARGIN_B),
        width=FIG_WIDTH,
        height=height,
    )
    return fig


# frames[(s, e)] = DataFrame indexed by day / home_id, one column per label


def plot_vs_epsilon(frames, eps_sorted, sens_sorted, labels, title, metric):
    """x = ε. Color = label, dash/marker = Δ. Mean ± std across days.
    With a single Δ the legend shows just the label (Δ goes in the title)."""
    fig = go.Figure()
    single = len(sens_sorted) == 1
    for i, label in enumerate(labels):
        color = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
        lbl = display_label(label)
        for j, s in enumerate(sens_sorted):
            means = np.array([frames[(s, e)][label].mean() for e in eps_sorted])
            stds = np.array([frames[(s, e)][label].std() for e in eps_sorted])
            group = f"{label}|{s:g}"
            fig.add_trace(
                go.Scatter(
                    x=eps_sorted,
                    y=means,
                    mode="lines+markers",
                    name=lbl
                    if (single and lbl)
                    else _legend(lbl, f"{DELTA}={s:g}"),
                    legendgroup=group,
                    marker=dict(
                        size=8, color=color, symbol=SYMBOLS[j % len(SYMBOLS)]
                    ),
                    line=dict(
                        width=2.5, color=color, dash=DASHES[j % len(DASHES)]
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=eps_sorted + eps_sorted[::-1],
                    y=list(means + stds) + list((means - stds)[::-1]),
                    fill="toself",
                    fillcolor=_hex_to_rgba(color, 0.08),
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=group,
                )
            )
    fig = _apply_layout(
        fig, title, EPS_AXIS_TITLE, METRIC_DISPLAY_NAMES.get(metric, metric)
    )
    fig.update_xaxes(tickmode="array", tickvals=eps_sorted)
    return fig


def plot_vs_sensitivity(frames, eps_sorted, sens_sorted, labels, title, metric):
    """x = Δ. Color = label, dash/marker = ε. Mean across days.
    With a single ε the legend shows just the label (ε goes in the title)."""
    fig = go.Figure()
    single = len(eps_sorted) == 1
    for i, label in enumerate(labels):
        color = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
        lbl = display_label(label)
        for j, e in enumerate(eps_sorted):
            means = [frames[(s, e)][label].mean() for s in sens_sorted]
            fig.add_trace(
                go.Scatter(
                    x=sens_sorted,
                    y=means,
                    mode="lines+markers",
                    name=lbl if (single and lbl) else _legend(lbl, f"ε={e:g}"),
                    legendgroup=f"{label}|{e:g}",
                    marker=dict(
                        size=8, color=color, symbol=SYMBOLS[j % len(SYMBOLS)]
                    ),
                    line=dict(
                        width=2.5, color=color, dash=DASHES[j % len(DASHES)]
                    ),
                )
            )
    fig = _apply_layout(
        fig, title, SENS_AXIS_TITLE, METRIC_DISPLAY_NAMES.get(metric, metric)
    )
    fig.update_xaxes(tickmode="array", tickvals=sens_sorted)
    return fig


def plot_vs_index(
    frames,
    eps_sorted,
    sens_sorted,
    label,
    title,
    metric,
    index_col="day",
    eps_subset=None,
    baseline_label=None,
):
    """x = row index (day or home_id), single label. Color = ε, dash/marker = Δ.
    Optional flat baseline (e.g. true_power_consumption), taken from the first
    available frame since it doesn't vary by ε or Δ. Households are drawn as
    unconnected markers, sorted by the baseline when one is available."""
    fig = go.Figure()
    # home ids have no order, so don't connect them with lines
    mode = "markers" if index_col == "home_id" else "lines+markers"

    # household order: ascending baseline, so the estimates scatter around a
    # smooth reference curve instead of a jagged one
    order = None
    if (
        index_col == "home_id"
        and SORT_HOMES_BY_BASELINE
        and baseline_label is not None
    ):
        for df in frames.values():
            if baseline_label in df.columns:
                order = list(df[baseline_label].sort_values().index)
                break

    def prep(df):
        if order is not None:
            df = df.reindex(order)
        if index_col == "home_id":
            df = df.copy()
            df.index = df.index.astype(str)  # ids are labels, not numbers
        return df

    if baseline_label is not None:
        for df in frames.values():
            if baseline_label in df.columns:
                df = prep(df)
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[baseline_label],
                        mode=mode,
                        name=display_label(baseline_label),
                        marker=dict(size=6, color="black", symbol="x"),
                        line=dict(width=2, color="black", dash="dash"),
                    )
                )
                break

    shown_eps = [e for e in eps_sorted if eps_subset is None or e in eps_subset]
    single_e = len(shown_eps) == 1
    single_s = len(sens_sorted) == 1
    for i, e in enumerate(shown_eps):
        for j, s in enumerate(sens_sorted):
            # when ε is fixed (Δ sweep) colour by Δ, otherwise colour by ε
            color = DISTINCT_COLORS[
                (j if single_e else i) % len(DISTINCT_COLORS)
            ]
            df = prep(frames[(s, e)])
            # the fixed parameter lives in the title; name lines by what varies
            parts = []
            if not single_e or single_s:
                parts.append(f"ε={e:g}")
            if not single_s or single_e:
                parts.append(f"{DELTA}={s:g}")
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[label],
                    mode=mode,
                    name=_legend(*parts),
                    marker=dict(
                        size=6, color=color, symbol=SYMBOLS[j % len(SYMBOLS)]
                    ),
                    line=dict(
                        width=2, color=color, dash=DASHES[j % len(DASHES)]
                    ),
                )
            )
    fig = _apply_layout(
        fig,
        title,
        INDEX_DISPLAY[index_col][0],
        METRIC_DISPLAY_NAMES.get(metric, metric),
        tickangle=-45,
    )
    if index_col == "home_id":
        axis = dict(type="category")  # ids are labels, not a numeric scale
        if order is not None:
            axis.update(
                categoryorder="array", categoryarray=[str(o) for o in order]
            )
        fig.update_xaxes(**axis)
    return fig


def heatmap_metric(
    frames,
    eps_sorted,
    sens_sorted,
    labels,
    title,
    metric,
    font_size=11,
    baseline_label=None,
):
    """rows = (label, Δ), cols = ε, color = mean across days.
    An optional baseline (true power) is added as ONE row at the top, since it
    doesn't vary with ε or Δ; it is taken from the first available frame."""
    row_names, z = [], []
    if baseline_label is not None:
        for df in frames.values():
            if baseline_label in df.columns:
                row_names.append(display_label(baseline_label))
                z.append([df[baseline_label].mean()] * len(eps_sorted))
                break
    for label in labels:
        for s in sens_sorted:
            row_names.append(_legend(display_label(label), f"{DELTA}={s:g}"))
            z.append([frames[(s, e)][label].mean() for e in eps_sorted])
    z = np.array(z, dtype=float)

    is_bias = metric == "bias"
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[f"{e:g}" for e in eps_sorted],
            y=row_names,
            colorscale="RdBu_r" if is_bias else "Viridis",
            zmid=0 if is_bias else None,
            colorbar=dict(title=METRIC_DISPLAY_NAMES.get(metric, metric)),
            text=[
                [f"{v:,.2f}" if not np.isnan(v) else "" for v in row]
                for row in z
            ],
            texttemplate="%{text}",
            textfont=dict(size=font_size),
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _apply_layout(
        fig,
        title,
        EPS_AXIS_TITLE,
        HEATMAP_Y_TITLE,
        height=max(FIG_HEIGHT, 28 * len(row_names) + 200),
    )


def save_fig(fig, name, metric, out: Output, sub=None):
    """Save to html/<metric>[/<sub>]/ and pdf/<metric>[/<sub>]/."""
    parts = (metric,) if sub is None else (metric, sub)
    html_metric_dir = out.html.joinpath(*parts)
    pdf_metric_dir = out.pdf.joinpath(*parts)
    html_metric_dir.mkdir(parents=True, exist_ok=True)
    pdf_metric_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_metric_dir / f"{name}.html")
    fig.write_image(pdf_metric_dir / f"{name}.pdf", scale=FIG_SCALE)  # kaleido


def make_figures(
    metric,
    frames,
    eps_sorted,
    sens_sorted,
    labels,
    title_suffix,
    out,
    index_col="day",
    day_view_labels=None,
    day_view_eps=None,
    title_tag=None,
):
    """ε sweep, Δ sweep, heatmap, and optional per-day / per-home diagnostics."""
    tag = None if metric == ENERGY_METRIC else title_tag
    title = display_title(metric, title_suffix, tag)
    base = f"{metric}_{title_suffix}"

    # vs_eps/: all Δ together, then all ε at each single Δ
    fig = plot_vs_epsilon(
        frames, eps_sorted, sens_sorted, labels, title, metric
    )
    save_fig(fig, f"{base}_vs_eps", metric, out, sub="vs_eps")
    for s in sens_sorted:
        fig = plot_vs_epsilon(
            frames, eps_sorted, [s], labels, f"{title} — {DELTA}={s:g}", metric
        )
        save_fig(fig, f"{base}_vs_eps_sens={s:g}", metric, out, sub="vs_eps")

    # vs_sens/: all ε together, then all Δ at each single ε
    fig = plot_vs_sensitivity(
        frames, eps_sorted, sens_sorted, labels, title, metric
    )
    save_fig(fig, f"{base}_vs_sens", metric, out, sub="vs_sens")
    for e in eps_sorted:
        fig = plot_vs_sensitivity(
            frames, [e], sens_sorted, labels, f"{title} — ε={e:g}", metric
        )
        save_fig(fig, f"{base}_vs_sens_eps={e:g}", metric, out, sub="vs_sens")

    # true power baseline only makes sense on the actual power totals table,
    # for the power branch — not on error-metric tables.
    baseline = (
        BASELINE_LABEL
        if metric == ENERGY_METRIC
        and title_suffix not in ("mains", "appliances")
        else None
    )

    fig = heatmap_metric(
        frames,
        eps_sorted,
        sens_sorted,
        labels,
        f"{title} — Heatmap",
        metric,
        baseline_label=baseline,
    )
    save_fig(fig, f"{base}_heatmap", metric, out)

    shown_eps = [
        e for e in eps_sorted if day_view_eps is None or e in day_view_eps
    ]
    _, title_part, file_part = INDEX_DISPLAY[index_col]
    for label in day_view_labels or []:
        if label not in labels:
            continue
        stem = f"{base}_{label}_{file_part}"
        # a blank label (billing) leaves just "<title> — by Home"
        head = f"{title} — " + " ".join(
            p for p in (display_label(label), title_part) if p
        )

        def day_fig(eps_list, sens_list, suffix=""):
            return plot_vs_index(
                frames,
                eps_list,
                sens_list,
                label,
                head + suffix,
                metric,
                index_col=index_col,
                eps_subset=day_view_eps,
                baseline_label=baseline,
            )

        # all ε and Δ together
        save_fig(day_fig(eps_sorted, sens_sorted), stem, metric, out)
        # one Δ at a time (lines = ε)
        for s in sens_sorted:
            fig = day_fig(eps_sorted, [s], f" — {DELTA}={s:g}")
            save_fig(fig, f"{stem}_sens={s:g}", metric, out)
        # one ε at a time (lines = Δ)
        for e in shown_eps:
            fig = day_fig([e], sens_sorted, f" — ε={e:g}")
            save_fig(fig, f"{stem}_eps={e:g}", metric, out)


def load_frames(subdir: str, metric: str, index_col: str):
    """Returns {(s, e): DataFrame indexed by index_col}, or None if a file or
    its index column is missing."""
    frames = {}
    for s in sensitivity:
        for e in eps:
            fp = (
                RESULT_DIR
                / f"eps={e:g}"
                / f"w={window_size:g}"
                / f"peak={peak:g}"
                / f"sens={s:g}"
                / subdir
                / f"{metric}.csv"
            )
            if not fp.exists():
                print(f"[skip] {subdir}/{metric}: missing {fp}")
                return None
            df = pv.read_csv(fp).to_pandas()
            if index_col not in df.columns:
                print(
                    f"[skip] {subdir}/{metric}: no '{index_col}' column in {fp}"
                )
                return None
            frames[(s, e)] = df.set_index(index_col)
    return frames


def process_source(source_name: str, subdir: str, index_col: str):
    out = Output(RESULT_DIR / "figures" / source_name).make()
    eps_sorted = sorted(eps)
    sens_sorted = sorted(sensitivity)

    # summary[(s, e)][label][metric] = mean across days
    summary = defaultdict(lambda: defaultdict(dict))

    for metric in METRIC_KEYS:
        frames = load_frames(subdir, metric, index_col)
        if frames is None:
            continue

        # accumulate per-label mean for the summary tables, regardless of branch
        for (s, e), df in frames.items():
            for label, val in df.mean().items():
                summary[(s, e)][label][metric] = val

        # only labels present in every (Δ, ε) table can be compared across them
        common = set.intersection(*(set(df.columns) for df in frames.values()))
        first_cols = list(next(iter(frames.values())).columns)

        kwargs = dict(
            frames=frames,
            eps_sorted=eps_sorted,
            sens_sorted=sens_sorted,
            out=out,
            index_col=index_col,
            day_view_labels=DAY_VIEW_LABELS,
            day_view_eps=DAY_VIEW_EPS,
            title_tag=SOURCE_TITLE_TAG.get(source_name),
        )

        # Every metric (scale-dependent or normalised, e.g. sMAPE or
        # RMSE/MAE) is split into the same comparison groups.
        groups = {
            "power": [c for c in POWER_LABELS if c in common],
            "mains": [c for c in MAINS_LABELS if c in common],
            "appliances": [c for c in APPLIANCES_LABELS if c in common],
        }

        made_any = False
        for suffix, cols in groups.items():
            if cols:
                make_figures(metric, labels=cols, title_suffix=suffix, **kwargs)
                made_any = True

        if made_any:
            grouped = {c for cols in groups.values() for c in cols}
            skipped = [
                c
                for c in first_cols
                if c in common
                and c not in grouped
                and c != BASELINE_LABEL  # baseline, drawn separately
            ]
            if skipped:
                print(
                    f"[warn] {subdir}/{metric}: not in any group, not plotted: {skipped}"
                )
        else:
            # label set doesn't match the known groups (e.g. billing). The
            # baseline is excluded from the labels so it is drawn exactly once
            # (as the heatmap row / black crosses), and every remaining column
            # gets a day / by-household view.
            cols = [
                c for c in first_cols if c in common and c != BASELINE_LABEL
            ]
            make_figures(
                metric,
                labels=cols,
                title_suffix="all",
                **{**kwargs, "day_view_labels": cols},
            )

    # one summary CSV per (Δ, ε): rows = label, columns = metric
    for (s, e), label_dict in summary.items():
        df = pd.DataFrame(label_dict).T  # outer keys -> columns, so .T flips it
        df.index = [table_label(l) for l in df.index]
        df.index.name = "label"
        df.columns = [METRIC_DISPLAY_NAMES.get(c, c) for c in df.columns]
        df.to_csv(out.tables / f"summary_sens={s:g}_eps={e:g}.csv")


for source_name, (subdir, index_col) in SOURCES.items():
    process_source(source_name, subdir, index_col)
