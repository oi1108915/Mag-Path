#!/usr/bin/env python3
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# GLOBAL PLOTTING STYLE
# ----------------------------
plt.rc('font', family='serif', size=22)

GPS_COLOR_NEUTRAL = "slategray"
GPS_COLOR_RIGHT   = "red"
GPS_COLOR_LEFT    = "gold"
GPS_COLOR_PATH    = "lightgray"
START_MARKER_COLOR = "limegreen"
END_MARKER_COLOR   = "dodgerblue"

# ----------------------------
# PATHS
# ----------------------------
OUT_DIR = "./"
os.makedirs(OUT_DIR, exist_ok=True)

SEGMENTS_CSV_PATH = "./L-R_segments.csv"

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def confusion_2x2_from_eval(eval_df: pd.DataFrame):
    """
    Confusion matrix for evaluated set (both decisive).
    Rows: GT (right, left)
    Cols: EM (right, left)
    """
    gt = eval_df["gt_label"].to_numpy()
    em = eval_df["em_label"].to_numpy()
    labels = ["right", "left"]
    mat = np.zeros((2, 2), dtype=int)
    for i, g in enumerate(labels):
        for j, e in enumerate(labels):
            mat[i, j] = int(np.sum((gt == g) & (em == e)))
    return labels, mat


def main():
    if not os.path.exists(SEGMENTS_CSV_PATH):
        raise FileNotFoundError(f"Missing segments CSV: {SEGMENTS_CSV_PATH}")

    seg_df = pd.read_csv(SEGMENTS_CSV_PATH)

    required = [
        "segment_id", "t0_s", "t1_s", "seg_len_s", "n_samples",
        "east_m", "north_m",
        "drv1", "drv2", "drv3", "drv4",
        "bp_right_mean", "bp_left_mean", "bp_ratio_right_over_left",
        "dlon", "gt_label",
        "em_label", "em_decisive", "gt_decisive", "eval_ok", "correct"
    ]
    missing = [c for c in required if c not in seg_df.columns]
    if missing:
        raise ValueError(f"Segments CSV missing required columns: {missing}")

    # Numeric coercions (safe)
    num_cols = [
        "segment_id", "t0_s", "t1_s", "seg_len_s", "n_samples",
        "east_m", "north_m",
        "drv1", "drv2", "drv3", "drv4",
        "bp_right_mean", "bp_left_mean", "bp_ratio_right_over_left",
        "dlon", "correct"
    ]
    for c in num_cols:
        seg_df[c] = pd.to_numeric(seg_df[c], errors="coerce")

    # Ensure stable order
    seg_plot = seg_df.sort_values("segment_id").copy()

    # ----------------------------
    # (Optional) Compute evaluation numbers in-memory (no saving)
    # ----------------------------
    eval_df = seg_df.dropna(subset=["correct"]).copy()
    eval_df = eval_df[eval_df["eval_ok"] == True].copy()

    total_segments = int(len(seg_df))
    evaluated_n = int(len(eval_df))
    overall_cond = float(np.mean(eval_df["correct"].astype(float))) if evaluated_n > 0 else np.nan
    cond_labels, cond_conf = confusion_2x2_from_eval(eval_df) if evaluated_n > 0 else (["right", "left"], np.zeros((2, 2), dtype=int))

    print("\n============================================================")
    print("SEGMENT-TABLE PROCESSING COMPLETE")
    print("============================================================")
    print(f"Segments loaded: {total_segments}")
    if evaluated_n > 0 and np.isfinite(overall_cond):
        print(f"Evaluated (both decisive): {evaluated_n}")
        print(f"Conditional accuracy: {overall_cond:.6f}")
        print(f"Confusion labels: {cond_labels}")
        print(f"Confusion matrix: {cond_conf.tolist()}")
    else:
        print("No evaluated segments (eval_ok==True) available for accuracy computation.")

    # ----------------------------
    # GPS overlay plot (meters)
    # Path: connect segment centroids in time order
    # ----------------------------
    fig, ax = plt.subplots(figsize=(16, 8))

    east = pd.to_numeric(seg_plot["east_m"], errors="coerce").to_numpy()
    north = pd.to_numeric(seg_plot["north_m"], errors="coerce").to_numpy()

    finite_mask = np.isfinite(east) & np.isfinite(north)
    if np.sum(finite_mask) < 2:
        raise ValueError("Not enough finite east_m/north_m points to plot GPS overlay.")

    east = east[finite_mask]
    north = north[finite_mask]
    em_labels = seg_plot["em_label"].to_numpy()[finite_mask]

    ax.plot(east, north, color=GPS_COLOR_PATH, linewidth=1.2, alpha=0.85, label="Path (segment centroids)")

    ax.scatter(east[0],  north[0],  s=400, color=START_MARKER_COLOR, edgecolors="k", linewidths=1.0, label="Start")
    ax.scatter(east[-1], north[-1], s=400, color=END_MARKER_COLOR,   edgecolors="k", linewidths=1.0, label="End")

    right_mask = (em_labels == "right")
    left_mask  = (em_labels == "left")
    neut_mask  = (em_labels == "neutral")

    if np.any(right_mask):
        ax.scatter(east[right_mask], north[right_mask],
                   s=300, marker=">", color=GPS_COLOR_RIGHT,
                   edgecolors="k", linewidths=0.6, label="Right")
    if np.any(left_mask):
        ax.scatter(east[left_mask], north[left_mask],
                   s=300, marker="<", color=GPS_COLOR_LEFT,
                   edgecolors="k", linewidths=0.6, label="Left")
    if np.any(neut_mask):
        ax.scatter(east[neut_mask], north[neut_mask],
                   s=220, marker="o", color=GPS_COLOR_NEUTRAL,
                   edgecolors="k", linewidths=0.5, alpha=0.75, label="Neutral")

    ax.set_xlabel("Local East (m)")
    ax.set_ylabel("Local North (m)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=20)

    out_gps = os.path.join(OUT_DIR, f"gps_em_segments_{RUN_TS}.pdf")
    fig.savefig(out_gps, dpi=600, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------
    # BP side-means time series
    # ----------------------------
    fig2, ax2 = plt.subplots(figsize=(16, 8))

    t0 = pd.to_numeric(seg_plot["t0_s"], errors="coerce").to_numpy()
    t1 = pd.to_numeric(seg_plot["t1_s"], errors="coerce").to_numpy()
    t_mid = 0.5 * (t0 + t1)

    finite_t = np.isfinite(t_mid)
    if np.sum(finite_t) < 2:
        raise ValueError("Not enough finite segment times to plot BP timeseries.")

    t_mid = t_mid[finite_t]
    t_mid0 = t_mid - float(t_mid[0])

    bp_r = pd.to_numeric(seg_plot["bp_right_mean"], errors="coerce").to_numpy()[finite_t]
    bp_l = pd.to_numeric(seg_plot["bp_left_mean"], errors="coerce").to_numpy()[finite_t]
    em_labels_t = seg_plot["em_label"].to_numpy()[finite_t]

    ax2.plot(t_mid0, bp_r, linewidth=2.0, label="Right Motors Mean")
    ax2.plot(t_mid0, bp_l, linewidth=2.0, label="Left Motors Mean")

    r_mask = (em_labels_t == "right")
    l_mask = (em_labels_t == "left")
    n_mask = (em_labels_t == "neutral")

    if np.any(r_mask):
        ax2.scatter(t_mid0[r_mask], bp_r[r_mask], s=70, marker="o",
                    edgecolors="k", linewidths=0.3, label="Right")
    if np.any(l_mask):
        ax2.scatter(t_mid0[l_mask], bp_l[l_mask], s=70, marker="o",
                    edgecolors="k", linewidths=0.3, label="Left")
    if np.any(n_mask):
        ax2.scatter(t_mid0[n_mask], 0.5 * (bp_r[n_mask] + bp_l[n_mask]),
                    s=70, marker="o", color=GPS_COLOR_NEUTRAL,
                    edgecolors="k", linewidths=0.3, alpha=0.75, label="Neutral")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Welch band-power per 1 s (a.u.)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="best", fontsize=20)

    out_bp = os.path.join(OUT_DIR, f"bp_right_vs_left_timeseries_{RUN_TS}.pdf")
    fig2.savefig(out_bp, dpi=600, bbox_inches="tight")
    plt.close(fig2)

    print("\n[Saved outputs]")
    print(f"  - GPS overlay PDF: {out_gps}")
    print(f"  - BP timeseries PDF: {out_bp}")


if __name__ == "__main__":
    main()
