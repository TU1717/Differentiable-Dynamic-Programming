import os
import numpy as np
import matplotlib.pyplot as plt

P = np.arange(1, 11)
datasets = ["Beauty", "VideoGames", "Gowalla", "Pinterest"]

recall = {
    "MF": {
        "Beauty":     [0.1293, 0.1323, 0.1341, 0.1369, 0.1385, 0.1396, 0.1403, 0.1418, 0.1417, 0.1429],
        "VideoGames": [0.1239, 0.1270, 0.1286, 0.1310, 0.1325, 0.1336, 0.1349, 0.1356, 0.1360, 0.1357],
        "Gowalla":    [0.1441, 0.1484, 0.1506, 0.1513, 0.1527, 0.1532, 0.1540, 0.1545, 0.1562, 0.1566],
        "Pinterest":  [0.1490, 0.1525, 0.1539, 0.1545, 0.1551, 0.1555, 0.1559, 0.1562, 0.1563, 0.1566],
    },
    "LightGCN": {
        "Beauty":     [0.1331, 0.1338, 0.1349, 0.1374, 0.1392, 0.1405, 0.1413, 0.1421, 0.1425, 0.1430],
        "VideoGames": [0.1275, 0.1292, 0.1303, 0.1315, 0.1334, 0.1340, 0.1351, 0.1356, 0.1365, 0.1367],
        "Gowalla":    [0.1505, 0.1528, 0.1537, 0.1546, 0.1552, 0.1553, 0.1556, 0.1566, 0.1572, 0.1580],
        "Pinterest":  [0.1530, 0.1550, 0.1553, 0.1560, 0.1563, 0.1564, 0.1565, 0.1568, 0.1569, 0.1572],
    },
    "XSimGCL": {
        "Beauty":     [0.1212, 0.1272, 0.1308, 0.1338, 0.1365, 0.1371, 0.1379, 0.1382, 0.1391, 0.1399],
        "VideoGames": [0.1198, 0.1249, 0.1269, 0.1298, 0.1312, 0.1328, 0.1331, 0.1343, 0.1352, 0.1358],
        "Gowalla":    [0.1330, 0.1421, 0.1453, 0.1479, 0.1489, 0.1503, 0.1509, 0.1520, 0.1531, 0.1543],
        "Pinterest":  [0.1537, 0.1578, 0.1590, 0.1596, 0.1598, 0.1598, 0.1598, 0.1599, 0.1597, 0.1600],
    },
}

talos = {
    "MF": {
        "Beauty": 0.1343,
        "VideoGames": 0.1280,
        "Gowalla": 0.1504,
        "Pinterest": 0.1514,
    },
    "LightGCN": {
        "Beauty": 0.1405,
        "VideoGames": 0.1331,
        "Gowalla": 0.1537,
        "Pinterest": 0.1553,
    },
    "XSimGCL": {
        "Beauty": 0.1380,
        "VideoGames": 0.1327,
        "Gowalla": 0.1506,
        "Pinterest": 0.1597,
    },
}

output_files = {
    "MF": "FIG_DP_FY_VS_TALOS_RECALL20_MF.png",
    "LightGCN": "FIG_DP_FY_VS_TALOS_RECALL20_LIGHTGCN.png",
    "XSimGCL": "FIG_DP_FY_VS_TALOS_RECALL20_XSIMGCL.png",
}

DIFFDP_COLOR = "#ff7f0e"
TALOS_COLOR = "#1f77b4"

def plot_recall_figure(backbone, output_path):
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.05), dpi=300)
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for i, (ax, dataset) in enumerate(zip(axes, datasets)):
        y = np.asarray(recall[backbone][dataset], dtype=float)
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

        all_y = np.append(y, baseline)
        y_min = all_y.min()
        y_max = all_y.max()
        span = y_max - y_min
        pad = max(span * 0.16, 0.00045)
        ax.set_ylim(y_min - pad, y_max + pad)

        ax.set_xlabel(r"$P$", fontsize=10)
        if i == 0:
            ax.set_ylabel("Recall@20", fontsize=10)

        ax.text(
            0.5, -0.31,
            f"{panel_labels[i]} {dataset}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )

        ax.grid(True, which="major", linestyle=":", linewidth=0.55, alpha=0.35)
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
    plot_recall_figure(
        backbone,
        os.path.join(out_dir, output_files[backbone]),
    )
