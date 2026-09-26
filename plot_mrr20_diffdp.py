import os
import numpy as np
import matplotlib.pyplot as plt

P = np.arange(1, 11)
datasets = ["Beauty", "VideoGames", "Gowalla", "Pinterest"]

mrr = {
    "MF": {
        "Beauty": [0.0408, 0.0417, 0.0424, 0.0433, 0.0441, 0.0445, 0.0448, 0.0452, 0.0450, 0.0454],
        "VideoGames": [0.0309, 0.0315, 0.0320, 0.0328, 0.0333, 0.0332, 0.0337, 0.0337, 0.0341, 0.0340],
        "Gowalla": [0.0333, 0.0347, 0.0354, 0.0357, 0.0362, 0.0365, 0.0366, 0.0368, 0.0372, 0.0373],
        "Pinterest": [0.0333, 0.0344, 0.0350, 0.0350, 0.0352, 0.0354, 0.0354, 0.0355, 0.0356, 0.0358],
    },
    "LightGCN": {
        "Beauty": [0.0418, 0.0423, 0.0427, 0.0434, 0.0442, 0.0448, 0.0449, 0.0452, 0.0452, 0.0454],
        "VideoGames": [0.0316, 0.0320, 0.0325, 0.0328, 0.0334, 0.0335, 0.0337, 0.0339, 0.0339, 0.0340],
        "Gowalla": [0.0355, 0.0361, 0.0365, 0.0368, 0.0369, 0.0370, 0.0371, 0.0374, 0.0376, 0.0377],
        "Pinterest": [0.0345, 0.0352, 0.0353, 0.0355, 0.0356, 0.0356, 0.0358, 0.0358, 0.0358, 0.0358],
    },
    "XSimGCL": {
        "Beauty": [0.0381, 0.0403, 0.0415, 0.0425, 0.0434, 0.0438, 0.0440, 0.0441, 0.0445, 0.0447],
        "VideoGames": [0.0291, 0.0307, 0.0312, 0.0321, 0.0327, 0.0331, 0.0334, 0.0337, 0.0339, 0.0340],
        "Gowalla": [0.0285, 0.0316, 0.0330, 0.0339, 0.0344, 0.0347, 0.0350, 0.0354, 0.0358, 0.0362],
        "Pinterest": [0.0350, 0.0362, 0.0368, 0.0368, 0.0369, 0.0369, 0.0369, 0.0370, 0.0369, 0.0372],
    },
}

talos = {
    "MF": {
        "Beauty": 0.0425,
        "VideoGames": 0.0322,
        "Gowalla": 0.0356,
        "Pinterest": 0.0348,
    },
    "LightGCN": {
        "Beauty": 0.0440,
        "VideoGames": 0.0334,
        "Gowalla": 0.0365,
        "Pinterest": 0.0358,
    },
    "XSimGCL": {
        "Beauty": 0.0438,
        "VideoGames": 0.0329,
        "Gowalla": 0.0346,
        "Pinterest": 0.0372,
    },
}

output_files = {
    "MF": "FIG_DP_FY_VS_TALOS_MRR20_MF.png",
    "LightGCN": "FIG_DP_FY_VS_TALOS_MRR20_LIGHTGCN.png",
    "XSimGCL": "FIG_DP_FY_VS_TALOS_MRR20_XSIMGCL.png",
}

DIFFDP_COLOR = "#ff7f0e"
TALOS_COLOR = "#1f77b4"

def plot_mrr20(backbone, output_path):
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.05), dpi=300)
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for i, (ax, dataset) in enumerate(zip(axes, datasets)):
        y = np.asarray(mrr[backbone][dataset], dtype=float)
        baseline = talos[backbone][dataset]

        ax.plot(
            P, y,
            color=DIFFDP_COLOR,
            marker="s",
            markersize=4.2,
            linewidth=1.8,
            label="DiffDP",
            zorder=3,
        )

        ax.axhline(
            baseline,
            color=TALOS_COLOR,
            linestyle="--",
            linewidth=1.6,
            label="Talos",
            zorder=2,
        )

        ax.set_xlim(0.7, 10.3)
        ax.set_xticks(P)
        ax.set_xlabel(r"$P$", fontsize=10)

        all_y = np.append(y, baseline)
        y_min = all_y.min()
        y_max = all_y.max()
        span = y_max - y_min
        pad = max(span * 0.16, 0.00025)
        ax.set_ylim(y_min - pad, y_max + pad)

        if i == 0:
            ax.set_ylabel("MRR@20", fontsize=10)

        ax.text(
            0.5, -0.31,
            f"{panel_labels[i]} {dataset}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )

        ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.35)
        ax.tick_params(axis="both", labelsize=8, length=3)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        ax.legend(
            loc="lower right",
            fontsize=7.5,
            frameon=True,
            borderpad=0.3,
            handlelength=1.6,
            handletextpad=0.5,
            labelspacing=0.25,
        )

    fig.subplots_adjust(
        left=0.065,
        right=0.992,
        top=0.965,
        bottom=0.26,
        wspace=0.28,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)

out_dir = "."

for backbone in ["MF", "LightGCN", "XSimGCL"]:
    plot_mrr20(
        backbone,
        os.path.join(out_dir, output_files[backbone]),
    )
