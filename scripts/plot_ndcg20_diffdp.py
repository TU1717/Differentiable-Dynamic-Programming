import os
import numpy as np
import matplotlib.pyplot as plt

P = np.arange(1, 11)
datasets = ["Beauty", "VideoGames", "Gowalla", "Pinterest"]

ndcg = {
    "MF": {
        "Beauty": [0.0730, 0.0747, 0.0759, 0.0775, 0.0785, 0.0793, 0.0796, 0.0804, 0.0802, 0.0808],
        "VideoGames": [0.0509, 0.0520, 0.0528, 0.0540, 0.0547, 0.0548, 0.0555, 0.0557, 0.0561, 0.0559],
        "Gowalla": [0.1105, 0.1143, 0.1164, 0.1172, 0.1185, 0.1192, 0.1197, 0.1201, 0.1215, 0.1218],
        "Pinterest": [0.0999, 0.1026, 0.1039, 0.1042, 0.1047, 0.1051, 0.1054, 0.1055, 0.1057, 0.1060],
    },
    "LightGCN": {
        "Beauty": [0.0750, 0.0757, 0.0763, 0.0776, 0.0787, 0.0797, 0.0799, 0.0803, 0.0805, 0.0808],
        "VideoGames": [0.0523, 0.0529, 0.0535, 0.0541, 0.0549, 0.0551, 0.0555, 0.0558, 0.0560, 0.0561],
        "Gowalla": [0.1164, 0.1183, 0.1193, 0.1202, 0.1206, 0.1208, 0.1212, 0.1220, 0.1226, 0.1230],
        "Pinterest": [0.1029, 0.1047, 0.1049, 0.1054, 0.1057, 0.1058, 0.1060, 0.1061, 0.1062, 0.1064],
    },
    "XSimGCL": {
        "Beauty": [0.0676, 0.0715, 0.0737, 0.0755, 0.0770, 0.0776, 0.0780, 0.0783, 0.0789, 0.0792],
        "VideoGames": [0.0486, 0.0510, 0.0518, 0.0531, 0.0539, 0.0546, 0.0549, 0.0553, 0.0557, 0.0559],
        "Gowalla": [0.0981, 0.1066, 0.1103, 0.1127, 0.1139, 0.1150, 0.1157, 0.1167, 0.1179, 0.1191],
        "Pinterest": [0.1039, 0.1070, 0.1083, 0.1086, 0.1088, 0.1089, 0.1088, 0.1088, 0.1091, 0.1093],
    },
}

talos = {
    "MF": {
        "Beauty": 0.0763,
        "VideoGames": 0.0528,
        "Gowalla": 0.1175,
        "Pinterest": 0.1029,
    },
    "LightGCN": {
        "Beauty": 0.0795,
        "VideoGames": 0.0549,
        "Gowalla": 0.1202,
        "Pinterest": 0.1056,
    },
    "XSimGCL": {
        "Beauty": 0.0782,
        "VideoGames": 0.0543,
        "Gowalla": 0.1154,
        "Pinterest": 0.1091,
    },
}

output_files = {
    "MF": "FIG_DP_FY_VS_TALOS_NDCG20_MF.png",
    "LightGCN": "FIG_DP_FY_VS_TALOS_NDCG20_LIGHTGCN.png",
    "XSimGCL": "FIG_DP_FY_VS_TALOS_NDCG20_XSIMGCL.png",
}

DIFFDP_COLOR = "#ff7f0e"
TALOS_COLOR = "#1f77b4"

def plot_ndcg20(backbone, output_path):
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.05), dpi=300)
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for i, (ax, dataset) in enumerate(zip(axes, datasets)):
        y = np.asarray(ndcg[backbone][dataset], dtype=float)
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
        pad = max(span * 0.16, 0.00035)
        ax.set_ylim(y_min - pad, y_max + pad)

        if i == 0:
            ax.set_ylabel("NDCG@20", fontsize=10)

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
    plot_ndcg20(
        backbone,
        os.path.join(out_dir, output_files[backbone]),
    )
