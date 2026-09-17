# %%
import sys

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pyarrow as pa

sys.path.insert(0, "/Users/lullaby/Projects/University/dp-comparison-hons")

from src.mechanisms.gai import gai_aggregate, gai_counts_to_power, gai_mechanism


# %%
def test_gai(numbers, epsilon, num_subintervals):
    boundary_list = np.linspace(0, 100, num_subintervals + 1)
    # print(boundary_list)

    # %%
    perturbed = gai_mechanism(numbers, boundary_list, epsilon).reshape(1, -1)
    # print(perturbed)
    # %%
    counts = gai_aggregate(
        perturbed, epsilon, num_subintervals + 1, boundary_list
    )
    # print(counts)
    # %%
    table = pa.table(
        {str(boundary_list[i]): counts[:, i] for i in range(counts.shape[1])}
    )
    reconstructed = gai_counts_to_power(table)
    return reconstructed


eps = [1, 2, 3, 4, 5, 6]
subints = [4, 5, 10, 25, 50]


def main():
    rng = np.random.default_rng(17)
    numbers = rng.integers(0, 101, size=1000)
    agg_nums = np.sum(numbers)
    fig = go.Figure()
    colors = px.colors.qualitative.Bold  # distinct, readable colors

    # print(numbers)
    for i, subintervals in enumerate(subints):
        totals = {}
        for epsilon in eps:
            results = [
                test_gai(numbers, epsilon, subintervals) for _ in range(30)
            ]
            sums = [np.sum(r) for r in results]
            agg_reconstructed = np.mean(sums)
            # print(agg_reconstructed, agg_nums)
            totals[epsilon] = float(agg_reconstructed)
        print(totals)

        fig.add_trace(
            go.Scatter(
                x=list(totals.keys()),
                y=list(totals.values()),
                mode="lines+markers",
                name=f"{subintervals} subintervals",
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=9, line=dict(width=1, color="white")),
            )
        )

    fig.add_hline(
        y=agg_nums,
        line_dash="dash",
        line_color="gray",
        line_width=2,
        annotation_text=f"true aggregate = {agg_nums}",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=dict(
            text="Reconstructed Aggregate vs Epsilon by Subinterval Count",
            font=dict(size=20),
        ),
        xaxis=dict(
            title="Epsilon (privacy budget)",
            title_font=dict(size=15),
            tickfont=dict(size=13),
            gridcolor="lightgray",
            showgrid=True,
        ),
        yaxis=dict(
            title="Mean Reconstructed Aggregate (30 trials)",
            title_font=dict(size=15),
            tickfont=dict(size=13),
            gridcolor="lightgray",
            showgrid=True,
        ),
        legend=dict(
            title="Subintervals",
            font=dict(size=12),
            bordercolor="lightgray",
            borderwidth=1,
        ),
        plot_bgcolor="white",
        width=1000,
        height=600,
        margin=dict(t=80, b=60, l=70, r=40),
    )

    fig.write_html("gai_reconstruction.html")
    fig.write_image(
        "gai_reconstruction_x.png", scale=2
    )  # requires: pip install -U kaleido

    fig.show()


if __name__ == "__main__":
    main()
