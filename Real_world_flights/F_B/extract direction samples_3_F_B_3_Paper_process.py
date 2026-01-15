#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ----------------------------
# GLOBAL PLOTTING STYLE
# ----------------------------
plt.rc('font', family='serif', size=22)

GPS_COLOR_NEUTRAL = "slategray"
GPS_COLOR_FORWARD = "red"
GPS_COLOR_BACK    = "gold"
GPS_COLOR_PATH    = "lightgray"
START_MARKER_COLOR = "limegreen"
END_MARKER_COLOR   = "dodgerblue"

# ----------------------------
# PATHS
# ----------------------------
OUT_DIR = "./"
os.makedirs(OUT_DIR, exist_ok=True)

SEGMENTS_TABLE_PATH = "./F-B_segments.csv"

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def fmt_float(x, digits=6) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float) and not np.isfinite(x):
        return "NA"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


def pct(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{100.0 * float(x):.2f}%"


def confusion_2x2_from_eval(eval_df: pd.DataFrame):
    """
    Confusion matrix for evaluated set (both decisive).
    Rows: GT (forward, backward)
    Cols: EM (forward, backward)
    """
    gt = eval_df["gt_label"].to_numpy()
    em = eval_df["em_label"].to_numpy()
    labels = ["forward", "backward"]
    mat = np.zeros((2, 2), dtype=int)
    for i, g in enumerate(labels):
        for j, e in enumerate(labels):
            mat[i, j] = int(np.sum((gt == g) & (em == e)))
    return labels, mat


def main():
    if not os.path.exists(SEGMENTS_TABLE_PATH):
        raise FileNotFoundError(f"Missing segments table: {SEGMENTS_TABLE_PATH}")

    seg_df = pd.read_csv(SEGMENTS_TABLE_PATH)

    required = [
        "segment_id", "t0_s", "t1_s", "seg_len_s", "n_samples",
        "bp_drv1", "bp_drv2", "bp_drv3", "bp_drv4",
        "bp_front_mean", "bp_rear_mean", "bp_ratio_front_over_rear",
        "em_label", "gt_label", "dlon",
        "em_decisive", "gt_decisive", "eval_ok", "correct",
        "east_m", "north_m",
        "t_start_s_used", "t_end_s_used",
        "fs_drv_hz", "step_s", "min_samples_per_seg",
        "dominance_frac", "eps_lon", "gps_rotate_deg", "earth_radius_m"
    ]
    missing = [c for c in required if c not in seg_df.columns]
    if missing:
        raise ValueError(f"Segments table missing required columns: {missing}")

    # Coerce numerics
    num_cols = [
        "segment_id", "t0_s", "t1_s", "seg_len_s", "n_samples",
        "bp_drv1", "bp_drv2", "bp_drv3", "bp_drv4",
        "bp_front_mean", "bp_rear_mean", "bp_ratio_front_over_rear",
        "dlon", "correct", "east_m", "north_m",
        "t_start_s_used", "t_end_s_used",
        "fs_drv_hz", "step_s", "min_samples_per_seg",
        "dominance_frac", "eps_lon", "gps_rotate_deg", "earth_radius_m"
    ]
    for c in num_cols:
        seg_df[c] = pd.to_numeric(seg_df[c], errors="coerce")

    # Ensure stable ordering
    seg_df.sort_values("segment_id", inplace=True)
    seg_df.reset_index(drop=True, inplace=True)

    # Recover parameters (for reporting)
    t0_use = float(seg_df["t_start_s_used"].iloc[0])
    t1_use = float(seg_df["t_end_s_used"].iloc[0])
    fs_drv = float(seg_df["fs_drv_hz"].iloc[0])
    step_s = float(seg_df["step_s"].iloc[0])
    seg_len_s = float(seg_df["seg_len_s"].iloc[0])
    min_samples = int(seg_df["min_samples_per_seg"].iloc[0])
    dominance_frac = float(seg_df["dominance_frac"].iloc[0])
    eps_lon = float(seg_df["eps_lon"].iloc[0])
    gps_rotate_deg = float(seg_df["gps_rotate_deg"].iloc[0])

    # ----------------------------
    # Accuracy metrics (computed, NOT saved)
    # ----------------------------
    eval_df = seg_df.dropna(subset=["correct"]).copy()
    eval_df = eval_df[eval_df["eval_ok"] == True].copy()

    total_segments = int(len(seg_df))
    em_decisive_n  = int(np.sum(seg_df["em_decisive"].astype(int)))
    gt_decisive_n  = int(np.sum(seg_df["gt_decisive"].astype(int)))
    evaluated_n    = int(len(eval_df))

    overall_cond = float(np.mean(eval_df["correct"].astype(float))) if evaluated_n > 0 else np.nan

    def acc_dir_cond(direction: str):
        sub = eval_df[eval_df["gt_label"] == direction]
        if len(sub) == 0:
            return np.nan, 0
        return float(np.mean(sub["correct"].astype(float))), int(len(sub))

    acc_f_cond, n_f_cond = acc_dir_cond("forward")
    acc_b_cond, n_b_cond = acc_dir_cond("backward")
    cond_labels, cond_conf = confusion_2x2_from_eval(eval_df) if evaluated_n > 0 else (["forward", "backward"], np.zeros((2, 2), dtype=int))

    gt_df = seg_df[seg_df["gt_decisive"] == True].copy()
    N_gt = int(len(gt_df))

    coverage_gt = float(np.mean(gt_df["em_decisive"].astype(float))) if N_gt > 0 else np.nan

    decided_gt_df = gt_df[gt_df["em_decisive"] == True].copy()
    N_dec_gt = int(len(decided_gt_df))
    acc_decided_gt = float(np.mean((decided_gt_df["em_label"] == decided_gt_df["gt_label"]).astype(float))) if N_dec_gt > 0 else np.nan

    correct_on_gt = int(np.sum((gt_df["em_label"] == gt_df["gt_label"]) & (gt_df["em_decisive"] == True)))
    acc_e2e_gt = (correct_on_gt / N_gt) if N_gt > 0 else np.nan

    # ----------------------------
    # GPS overlay PDF (meters) from segment centroids
    # ----------------------------
    fig, ax = plt.subplots(figsize=(16, 8))

    east = pd.to_numeric(seg_df["east_m"], errors="coerce").to_numpy()
    north = pd.to_numeric(seg_df["north_m"], errors="coerce").to_numpy()
    em_labels = seg_df["em_label"].to_numpy()

    finite = np.isfinite(east) & np.isfinite(north)
    if np.sum(finite) < 2:
        raise ValueError("Not enough finite east_m/north_m points to plot overlay.")

    east = east[finite]
    north = north[finite]
    em_labels = em_labels[finite]

    ax.plot(east, north, color=GPS_COLOR_PATH, linewidth=1.2, alpha=0.85, label="Trajectory")

    ax.scatter(east[0],  north[0],  s=400, color=START_MARKER_COLOR, edgecolors="k", linewidths=1.0, label="Start")
    ax.scatter(east[-1], north[-1], s=400, color=END_MARKER_COLOR,   edgecolors="k", linewidths=1.0, label="End")

    f_mask = (em_labels == "forward")
    b_mask = (em_labels == "backward")
    n_mask = (em_labels == "neutral")

    if np.any(f_mask):
        ax.scatter(east[f_mask], north[f_mask], s=300, marker=">", color=GPS_COLOR_FORWARD,
                   edgecolors="k", linewidths=0.6, label="Forward")
    if np.any(b_mask):
        ax.scatter(east[b_mask], north[b_mask], s=300, marker="<", color=GPS_COLOR_BACK,
                   edgecolors="k", linewidths=0.6, label="Backward")
    if np.any(n_mask):
        ax.scatter(east[n_mask], north[n_mask], s=220, marker="o", color=GPS_COLOR_NEUTRAL,
                   edgecolors="k", linewidths=0.5, alpha=0.75, label="Neutral")

    ax.set_xlabel("Local East (m)")
    ax.set_ylabel("Local North (m)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=20)

    out_gps = os.path.join(OUT_DIR, f"{'F-B'}_gps_em_segments_{RUN_TS}.pdf")
    fig.savefig(out_gps, dpi=600, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------
    # BP front vs rear timeseries
    # ----------------------------
    seg_plot_df = seg_df.copy()
    seg_plot_df["t_mid_s"] = 0.5 * (seg_plot_df["t0_s"] + seg_plot_df["t1_s"])
    seg_plot_df["t_mid_s0"] = seg_plot_df["t_mid_s"] - float(seg_plot_df["t_mid_s"].iloc[0])
    seg_plot_df = seg_plot_df.dropna(subset=["bp_front_mean", "bp_rear_mean", "t_mid_s"]).copy()

    fig2, ax2 = plt.subplots(figsize=(16, 8))

    ax2.plot(
        seg_plot_df["t_mid_s0"].to_numpy(),
        seg_plot_df["bp_front_mean"].to_numpy(),
        linewidth=2.0,
        label="Front Motors Mean"
    )
    ax2.plot(
        seg_plot_df["t_mid_s0"].to_numpy(),
        seg_plot_df["bp_rear_mean"].to_numpy(),
        linewidth=2.0,
        label="Rear Motors Mean"
    )

    f_mask2 = (seg_plot_df["em_label"] == "forward").to_numpy()
    b_mask2 = (seg_plot_df["em_label"] == "backward").to_numpy()
    n_mask2 = (seg_plot_df["em_label"] == "neutral").to_numpy()

    ax2.scatter(seg_plot_df.loc[f_mask2, "t_mid_s0"], seg_plot_df.loc[f_mask2, "bp_front_mean"],
                s=70, marker="o", edgecolors="k", linewidths=0.3, label="Forward")
    ax2.scatter(seg_plot_df.loc[b_mask2, "t_mid_s0"], seg_plot_df.loc[b_mask2, "bp_rear_mean"],
                s=70, marker="o", edgecolors="k", linewidths=0.3, label="Backward")

    ax2.scatter(
        seg_plot_df.loc[n_mask2, "t_mid_s0"],
        0.5 * (seg_plot_df.loc[n_mask2, "bp_front_mean"] + seg_plot_df.loc[n_mask2, "bp_rear_mean"]),
        s=70, marker="o", color=GPS_COLOR_NEUTRAL,
        edgecolors="k", linewidths=0.3, alpha=0.75,
        label="Neutral"
    )

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Welch band-power per 1 s (a.u.)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="best", fontsize=20)

    out_bp = os.path.join(OUT_DIR, f"{'F-B'}_bp_front_vs_rear_timeseries_{RUN_TS}.pdf")
    fig2.savefig(out_bp, dpi=600, bbox_inches="tight")
    plt.close(fig2)

    # ----------------------------
    # Console summary
    # ----------------------------
    print("\n============================================================")
    print("RUN COMPLETE: Metrics + figures regenerated from segment table")
    print("============================================================\n")
    print("[Saved outputs]")
    print(f"  - GPS overlay PDF: {out_gps}")
    print(f"  - BP timeseries PDF: {out_bp}")
    print("")
    print("[Recovered parameters from table]")
    print(f"  Used window: [{t0_use:.3f}, {t1_use:.3f}] s")
    print(f"  fs_drv={fs_drv:.1f} Hz, seg_len={seg_len_s:.2f} s, step={step_s:.2f} s, min_samples={min_samples}")
    print(f"  δ={dominance_frac:.3f}, ε={eps_lon:g}, gps_rotation_deg={gps_rotate_deg:.1f}")
    print("")
    print("[EVAL A: Conditional (EM & GT decisive only)]")
    print(f"  Evaluated: {evaluated_n}/{total_segments} ({pct(evaluated_n/total_segments)})")
    print(f"  Overall:   {fmt_float(overall_cond)} ({pct(overall_cond)})")
    print(f"  |GT=FWD:   {fmt_float(acc_f_cond)} ({pct(acc_f_cond)}) with n={n_f_cond}")
    print(f"  |GT=BWD:   {fmt_float(acc_b_cond)} ({pct(acc_b_cond)}) with n={n_b_cond}")
    print(f"  Confusion (rows=GT, cols=EM) labels={cond_labels}: {cond_conf.tolist()}")
    print("")
    print("[EVAL B: GT-anchored (all GT decisive segments)]")
    print(f"  GT decisive: {N_gt}/{total_segments} ({pct(N_gt/total_segments)})")
    print(f"  Coverage on GT: {fmt_float(coverage_gt)} ({pct(coverage_gt)}) = {N_dec_gt}/{N_gt}")
    print(f"  Accuracy (decided-only on GT): {fmt_float(acc_decided_gt)} ({pct(acc_decided_gt)})")
    print(f"  Strict end-to-end accuracy on GT: {fmt_float(acc_e2e_gt)} ({pct(acc_e2e_gt)})")
    print("")


if __name__ == "__main__":
    main()
