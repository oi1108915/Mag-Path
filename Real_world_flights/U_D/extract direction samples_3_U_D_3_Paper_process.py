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
COLOR_UP          = "red"
COLOR_DOWN        = "gold"

OUT_DIR = "./"
os.makedirs(OUT_DIR, exist_ok=True)

SEGMENTS_TABLE_PATH = "./U-D-90_segments.csv"
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
    gt = eval_df["gt_label"].to_numpy()
    em = eval_df["em_label"].to_numpy()
    labels = ["up", "down"]
    mat = np.zeros((2, 2), dtype=int)
    for i, g in enumerate(labels):
        for j, e in enumerate(labels):
            mat[i, j] = int(np.sum((gt == g) & (em == e)))
    return labels, mat


def main():
    if not os.path.exists(SEGMENTS_TABLE_PATH):
        raise FileNotFoundError(f"Missing segments table: {SEGMENTS_TABLE_PATH}")

    seg_df = pd.read_csv(SEGMENTS_TABLE_PATH)

    # Find the BP column (expects bp_drv)
    bp_cols = [c for c in seg_df.columns if c.startswith("bp_")]
    if "bp_drv" not in seg_df.columns:
        raise ValueError(f"Segments table must contain 'bp_drv'. Found bp columns: {bp_cols}")

    required = [
        "segment_id", "t0_s", "t1_s", "seg_len_s", "n_samples",
        "bp_drv", "alt_mean_aligned_m", "alt_delta_next_m",
        "em_label", "gt_label", "em_decisive", "gt_decisive", "eval_ok", "correct",
        "t_start_s_used", "t_end_s_used",
        "fs_drv_hz", "step_s", "min_samples_per_seg",
        "alt_lag_s", "alt_delta_threshold_m", "bp_up_threshold",
    ]
    missing = [c for c in required if c not in seg_df.columns]
    if missing:
        raise ValueError(f"Segments table missing required columns: {missing}")

    # Coerce numerics
    num_cols = [
        "segment_id", "t0_s", "t1_s", "seg_len_s", "n_samples",
        "bp_drv", "alt_mean_aligned_m", "alt_delta_next_m",
        "t_start_s_used", "t_end_s_used",
        "fs_drv_hz", "step_s", "min_samples_per_seg",
        "alt_lag_s", "alt_delta_threshold_m", "bp_up_threshold",
        "correct",
    ]
    for c in num_cols:
        seg_df[c] = pd.to_numeric(seg_df[c], errors="coerce")

    # stable ordering
    seg_df.sort_values("segment_id", inplace=True)
    seg_df.reset_index(drop=True, inplace=True)

    # Recover parameters for reporting
    t0_use = float(seg_df["t_start_s_used"].iloc[0])
    t1_use = float(seg_df["t_end_s_used"].iloc[0])
    fs_drv = float(seg_df["fs_drv_hz"].iloc[0])
    step_s = float(seg_df["step_s"].iloc[0])
    seg_len_s = float(seg_df["seg_len_s"].iloc[0])
    min_samples = int(seg_df["min_samples_per_seg"].iloc[0])
    alt_lag_s = float(seg_df["alt_lag_s"].iloc[0])
    alt_thr = float(seg_df["alt_delta_threshold_m"].iloc[0])
    bp_thr = float(seg_df["bp_up_threshold"].iloc[0])

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

    acc_u_cond, n_u_cond = acc_dir_cond("up")
    acc_d_cond, n_d_cond = acc_dir_cond("down")

    cond_labels, cond_conf = confusion_2x2_from_eval(eval_df) if evaluated_n > 0 else (["up", "down"], np.zeros((2, 2), dtype=int))

    gt_df = seg_df[seg_df["gt_decisive"] == True].copy()
    N_gt = int(len(gt_df))
    coverage_gt = float(np.mean(gt_df["em_decisive"].astype(float))) if N_gt > 0 else np.nan

    decided_gt_df = gt_df[gt_df["em_decisive"] == True].copy()
    N_dec_gt = int(len(decided_gt_df))
    acc_decided_gt = float(np.mean((decided_gt_df["em_label"] == decided_gt_df["gt_label"]).astype(float))) if N_dec_gt > 0 else np.nan

    correct_on_gt = int(np.sum((gt_df["em_label"] == gt_df["gt_label"]) & (gt_df["em_decisive"] == True)))
    acc_e2e_gt = (correct_on_gt / N_gt) if N_gt > 0 else np.nan
    gt_em_neutral = int(np.sum((gt_df["em_decisive"] == False)))

    # ----------------------------
    # FIGURE: Time-series across segments (alt mean + BP), x normalized to start at 0
    # ----------------------------
    seg_plot_df = seg_df.copy()
    seg_plot_df["t_mid_s"]  = 0.5 * (seg_plot_df["t0_s"] + seg_plot_df["t1_s"])
    seg_plot_df["t_mid_s0"] = seg_plot_df["t_mid_s"] - float(seg_plot_df["t_mid_s"].iloc[0])
    seg_plot_df.sort_values("t_mid_s0", inplace=True)
    seg_plot_df.reset_index(drop=True, inplace=True)

    x   = seg_plot_df["t_mid_s0"].to_numpy(dtype=float)
    alt = seg_plot_df["alt_mean_aligned_m"].to_numpy(dtype=float)
    bp  = seg_plot_df["bp_drv"].to_numpy(dtype=float)
    em  = seg_plot_df["em_label"].astype(str).to_numpy()

    fig, ax1 = plt.subplots(figsize=(16, 8))

    ax1.plot(x, alt, linewidth=2.4, label="Altitude")
    ax1.scatter(x, alt, s=55, edgecolors="k", linewidths=0.3, label="_nolegend_")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Altitude (m)")
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(x, bp, linewidth=2.0, label="DRV band-power")
    ax2.axhline(float(bp_thr), linestyle="--", linewidth=2.2, alpha=0.85, label="Threshold")
    ax2.set_ylabel("Welch band-power (a.u.)")

    up_mask = (em == "up") & np.isfinite(bp) & np.isfinite(x)
    dn_mask = (em == "down") & np.isfinite(bp) & np.isfinite(x)
    nt_mask = (em == "neutral") & np.isfinite(bp) & np.isfinite(x)

    if np.any(up_mask):
        ax2.scatter(x[up_mask], bp[up_mask], s=150, marker="^",
                    color=COLOR_UP, edgecolors="k", linewidths=0.45, label="Up")
    if np.any(dn_mask):
        ax2.scatter(x[dn_mask], bp[dn_mask], s=150, marker="v",
                    color=COLOR_DOWN, edgecolors="k", linewidths=0.45, label="Down")
    if np.any(nt_mask):
        ax2.scatter(x[nt_mask], bp[nt_mask], s=95, marker="o",
                    color=GPS_COLOR_NEUTRAL, edgecolors="k", linewidths=0.35, alpha=0.80, label="Neutral")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=20)

    out_ts = os.path.join(OUT_DIR, f"U-D_alt_vs_drvbp_timeseries_{RUN_TS}.pdf")
    fig.savefig(out_ts, dpi=600, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------
    # Console summary
    # ----------------------------
    print("\n============================================================")
    print("RUN COMPLETE: Metrics + time-series regenerated from segment table")
    print("============================================================\n")
    print("[Saved outputs]")
    print(f"  - Time-series PDF: {out_ts}")
    print("")
    print("[Recovered parameters from table]")
    print(f"  Used window: [{t0_use:.3f}, {t1_use:.3f}] s")
    print(f"  fs_drv={fs_drv:.1f} Hz, seg_len={seg_len_s:.2f} s, step={step_s:.2f} s, min_samples={min_samples}")
    print(f"  alt_lag_s={alt_lag_s:g}, alt_delta_threshold_m={alt_thr:g}, bp_up_threshold={bp_thr:g}")
    print("")
    print("[EVAL A: Conditional (EM & GT decisive only)]")
    print(f"  Evaluated: {evaluated_n}/{total_segments} ({pct(evaluated_n/total_segments)})")
    print(f"  Overall:   {fmt_float(overall_cond)} ({pct(overall_cond)})")
    print(f"  |GT=UP:    {fmt_float(acc_u_cond)} ({pct(acc_u_cond)}) with n={n_u_cond}")
    print(f"  |GT=DOWN:  {fmt_float(acc_d_cond)} ({pct(acc_d_cond)}) with n={n_d_cond}")
    print(f"  Confusion (rows=GT, cols=EM) labels={cond_labels}: {cond_conf.tolist()}")
    print("")
    print("[EVAL B: GT-anchored (all GT decisive segments)]")
    print(f"  GT decisive: {N_gt}/{total_segments} ({pct(N_gt/total_segments)})")
    print(f"  Coverage on GT: {fmt_float(coverage_gt)} ({pct(coverage_gt)}) = {N_dec_gt}/{N_gt}")
    print(f"  Accuracy (decided-only on GT): {fmt_float(acc_decided_gt)} ({pct(acc_decided_gt)})")
    print(f"  Strict end-to-end accuracy on GT: {fmt_float(acc_e2e_gt)} ({pct(acc_e2e_gt)})")
    print(f"  EM undecided within GT-decisive: {gt_em_neutral}")
    print("")


if __name__ == "__main__":
    main()
