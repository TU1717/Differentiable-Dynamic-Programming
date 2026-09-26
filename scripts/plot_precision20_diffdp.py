import os
import numpy as np
import matplotlib.pyplot as plt

P = np.arange(1, 11)
datasets = ["Beauty", "VideoGames", "Gowalla", "Pinterest"]

precision = {
    "MF": {
        "Beauty": [0.0152, 0.0156, 0.0158, 0.0161, 0.0163, 0.0164, 0.0165, 0.0168, 0.0168, 0.0170],
        "VideoGames": [0.0062, 0.0064, 0.0064, 0.0066, 0.0066, 0.0067, 0.0067, 0.0068, 0.0068, 0.0068],
        "Gowalla": [0.0433, 0.0448, 0.0456, 0.0459, 0.0464, 0.0466, 0.0469, 0.0470, 0.0476, 0.0476],
        "Pinterest": [0.0392, 0.0401, 0.0405, 0.0406, 0.0408, 0.0409, 0.0410, 0.0410, 0.0411, 0.0412],
    },
    "LightGCN": {
        "Beauty": [0.0157, 0.0158, 0.0159, 0.0162, 0.0164, 0.0166, 0.0167, 0.0168, 0.0169, 0.0170],
        "VideoGames": [0.0064, 0.0065, 0.0065, 0.0066, 0.0067, 0.0067, 0.0068, 0.0068, 0.0068, 0.0068],
        "Gowalla": [0.0459, 0.0466, 0.0469, 0.0472, 0.0474, 0.0474, 0.0475, 0.0478, 0.0481, 0.0482],
        "Pinterest": [0.0402, 0.0407, 0.0408, 0.0410, 0.0411, 0.0411, 0.0412, 0.0412, 0.0412, 0.0413],
    },
    "XSimGCL": {
        "Beauty": [0.0147, 0.0153, 0.0157, 0.0160, 0.0163, 0.0164, 0.0165, 0.0165, 0.0167, 0.0168],
        "VideoGames": [0.0060, 0.0062, 0.0063, 0.0065, 0.0066, 0.0066, 0.0067, 0.0067, 0.0068, 0.0068],
        "Gowalla": [0.0404, 0.0432, 0.0442, 0.0450, 0.0453, 0.0457, 0.0460, 0.0464, 0.0467, 0.0471],
        "Pinterest": [0.0405, 0.0415, 0.0417, 0.0419, 0.0420, 0.0422, 0.0421, 0.0420, 0.0420, 0.0422],
    },
}

talos = {
    "MF": {
        "Beauty": 0.0157,
        "VideoGames": 0.0064,
        "Gowalla": 0.0452,
        "Pinterest": 0.0398,
    },
    "LightGCN": {
        "Beauty": 0.0164,
        "VideoGames": 0.0067,
        "Gowalla": 0.0464,
        "Pinterest": 0.0409,
    },
    "XSimGCL": {
        "Beauty": 0.0162,
        "VideoGames": 0.0066,
        "Gowalla": 0.0454,
        "Pinterest": 0.0421,
    },
}

output_files = {
    "MF": "FIG_DP_FY_VS_TALOS_PRECISION20_MF.png",
    "LightGCN": "FIG_DP_FY_VS_TALOS_PRECISION20_LIGHTGCN.png",
    "XSimGCL": "FIG_DP_FY_VS_TALOS_PRECISION20_XSIMGCL.png",
}

DIFFDP_COLOR = "#ff7f0e"
TALOS_COLOR = "#1f77b4"

def plot_precision20(backbone, output_path):
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.05), dpi=300)
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for i, (ax, dataset) in enumerate(zip(axes, datasets)):
        y = np.asarray(precision[backbone][dataset], dtype=float)
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
        pad = max(span * 0.16, 0.00008)
        ax.set_ylim(y_min - pad, y_max + pad)

        if i == 0:
            ax.set_ylabel("Precision@20", fontsize=10)

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
    plot_precision20(
        backbone,
        os.path.join(out_dir, output_files[backbone]),
    )
