#!/usr/bin/env python3
import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch, detrend

# Set the global font properties
font = {'family': 'serif', 'weight': 'normal', 'size': 30}
plt.rc('font', **font)

# ========= PWM categorical axis (for combined plots) =========
PWM_TICK_LABELS = [
    "OFF", "ON",
    "7", "10", "13", "17", "20", "23", "27", "30",
    "33", "37", "40", "43", "47", "50", "53", "57",
    "60", "63", "67", "70", "73", "77", "80", "83",
    "87", "90", "93", "97", "100"
]

RMS_XLABEL = "PWM (%)"
PSD_XLABEL = "PWM (%)"

# ========= CONFIG =========
folder_path   = "./"   # folder with CSV
reports_root  = "./reports_em_data_DRV"

save_plots    = True
show_plots    = False
dpi_out       = 600

# Visual consistency
vmin_db, vmax_db = -100, -20
fmax_hz      = 50          # y-limit for spectrograms & PSDs (None = auto)
win_sec      = 2.0         # spectrogram window length (sec)
overlap      = 0.5         # spectrogram overlap fraction
welch_win    = 4.0         # Welch window (sec)
welch_ovl    = 0.5         # Welch overlap
rms_win      = 2.0         # short-time RMS window (sec)
rms_ovl      = 0.5

# Smoothing controls
rms_smooth_pts    = 7      # moving-average window (points) for RMS smoothing
psd_smooth_bins   = 11     # moving-average window (bins) for PSD smoothing (odd recommended)

# Continuous concatenation gaps
cont_gap_time_s   = 3.0    # horizontal gap between files (RMS continuous & spec concatenation)
cont_gap_freq_hz  = 5.0    # horizontal gap between files (PSD continuous)

# Marker styling for "dots connected by line"
line_marker = 'o'
line_markersize = 4

# ========= CORE UTILS =========
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def infer_fs_from_time_ms(time_ms: np.ndarray) -> float:
    t = pd.to_numeric(pd.Series(time_ms), errors="coerce").to_numpy()
    if np.isnan(t).any():
        raise ValueError("Time column has non-numeric values.")
    if t.size < 2:
        raise ValueError("Not enough samples to infer fs.")
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Non-positive or invalid Δt.")
    return 1000.0 / np.median(dt)  # Hz

def compute_spec_fixed(sig, fs, win_s, ovl):
    nperseg  = max(32, int(round(fs * win_s)))
    noverlap = min(nperseg - 1, int(round(nperseg * ovl)))
    f, t, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, 10 * np.log10(Sxx + 1e-10)

def compute_welch(sig, fs, win_s, ovl):
    nperseg  = max(32, int(round(fs * win_s)))
    noverlap = min(nperseg - 1, int(round(nperseg * ovl)))
    f, Pxx = welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, 10 * np.log10(Pxx + 1e-20)

def short_time_rms(sig, fs, win_s, ovl):
    n = max(8, int(round(fs * win_s)))
    step = max(1, int(round(n * (1 - ovl))))
    idx0 = np.arange(0, len(sig) - n + 1, step)
    if idx0.size == 0:
        return np.array([]), np.array([])
    windows = sig[idx0[:, None] + np.arange(n)]
    rms = np.sqrt(np.mean(windows**2, axis=1))
    t = (idx0 + n/2) / fs
    return t, rms

def moving_average(y, win):
    win = max(1, int(win))
    if win == 1 or y.size == 0:
        return y
    k = np.ones(win, dtype=float) / win
    return np.convolve(y, k, mode="same")

def smooth_psd(Pdb, bins):
    return moving_average(Pdb, bins)

# ========= PWM sort helper =========
def pwm_sort_key(name: str):
    """
    Sort order:
      OFF -> ON -> numeric PWM ascending
    Works with base names like: "00_OFF", "0_ON", "7", "13", "20", ..., "100"
    """
    s = str(name).upper()
    if "OFF" in s:
        return (0, 0)
    if "ON" in s:
        return (1, 0)
    m = re.search(r'(\d+)', s)
    if m:
        return (2, int(m.group(1)))
    return (9, 0)

# ========= COLORS (KEEP YOUR SCHEME AS-IS) =========
# Your last active palette in the provided script is the red gradient "palette30".
palette30 = [
    "#F19CA0","#EE8990","#EB767F","#E8636F","#E3575E","#DD5259","#D74E54",
    "#D34B50","#CE474C","#C84347","#C23F43","#BC3B3F","#B6373B","#B03337",
    "#AA3033","#A42C30","#9E282C","#982428","#922023","#961E27","#8E1C24",
    "#861A22","#7E181F","#76161D","#6E131A","#661117","#5E0F14","#560C12",
    "#4E090F","#46070C","#3E0508"
]
def pick_color(idx: int) -> str:
    return palette30[idx % len(palette30)]

combined = {
    "rms":  {i: [] for i in range(4)},
    "psd":  {i: [] for i in range(4)},
    "spec": {i: [] for i in range(4)}
}

# ========= TRIMMING HELPERS (UNIFORM LENGTH ACROSS FILES) =========
def prescan_min_duration_s(files):
    """Return minimal duration (seconds) among CSVs that have at least 2 columns & 2 time samples."""
    mins = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            if df.shape[1] < 2:
                continue
            t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
            t = t[np.isfinite(t)]
            if t.size < 2:
                continue
            dur_s = (t[-1] - t[0]) / 1000.0
            if dur_s > 0:
                mins.append(dur_s)
        except Exception:
            continue
    return min(mins) if mins else None

def trim_by_duration(time_ms, arrays, duration_s):
    """
    Trim time_ms and each array in `arrays` to [0, duration_s] relative to the first time.
    arrays: list of np.ndarrays with same length as time_ms.
    """
    if duration_s is None or len(time_ms) == 0:
        return time_ms, arrays
    rel_t = (time_ms - time_ms[0]) / 1000.0
    end_idx = np.searchsorted(rel_t, duration_s, side='right')
    end_idx = max(1, min(end_idx, len(time_ms)))
    trimmed_time = time_ms[:end_idx]
    trimmed_arrays = [a[:end_idx] if a is not None and len(a) >= end_idx else a for a in arrays]
    return trimmed_time, trimmed_arrays

# ========= PER-FILE LOADER (NO PER-FILE PLOTS; ONLY POPULATE combined) =========
def process_file_to_combined(csv_path: str, uniform_duration_s: float = None):
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        print(f"[SKIP] {os.path.basename(csv_path)} needs at least 2 columns (time_ms + ≥1 motor).")
        return

    time_ms_all = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    if not np.isfinite(time_ms_all).all():
        mask = np.isfinite(time_ms_all)
        df = df.loc[mask].reset_index(drop=True)
        time_ms_all = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()

    num_motors = min(4, df.shape[1] - 1)
    motors_all = [pd.to_numeric(df.iloc[:, i], errors="coerce").to_numpy()
                  for i in range(1, 1 + num_motors)]
    motors_all = [np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0) for m in motors_all]

    # Trim to uniform shortest duration across ALL files
    time_ms, motors_all = trim_by_duration(time_ms_all, motors_all, uniform_duration_s)

    if len(time_ms) < 2:
        print(f"[SKIP] {os.path.basename(csv_path)} has <2 usable time samples after trimming.")
        return

    # Detrend after trimming
    motors = [detrend(m, type='constant') for m in motors_all]

    # Use inferred fs (or hardcode if you truly want fs=250)
    fs = infer_fs_from_time_ms(time_ms)

    base = os.path.splitext(os.path.basename(csv_path))[0]
    dur_s = (time_ms[-1] - time_ms[0]) / 1000.0
    print(f"[OK] {base}  fs≈{fs:.6f} Hz  dur≈{dur_s:.3f}s")

    for i in range(num_motors):
        # Spectrogram
        f, t, Sdb = compute_spec_fixed(motors[i], fs, win_sec, overlap)
        combined["spec"][i].append({"f": f, "t": t, "Sdb": Sdb, "file": csv_path, "base": base})

        # PSD
        f_psd, Pdb = compute_welch(motors[i], fs, welch_win, welch_ovl)
        Pdb_s = smooth_psd(Pdb, psd_smooth_bins)
        combined["psd"][i].append({"f": f_psd, "y": Pdb, "y_s": Pdb_s, "file": csv_path, "base": base, "fs": fs})

        # RMS
        t_rms, rms = short_time_rms(motors[i], fs, rms_win, rms_ovl)
        rms_s = moving_average(rms, rms_smooth_pts)
        combined["rms"][i].append({"t": t_rms, "y": rms, "y_s": rms_s, "file": csv_path, "base": base})

# ========= COMBINED PLOTS ONLY (PER MOTOR) =========
def combined_plots_all_files(out_root):
    if save_plots:
        ensure_dir(out_root)

    # -------- Per motor: RMS continuous (original + smoothed) with PWM categorical x-axis --------
    for i in range(4):
        entries = combined["rms"][i]
        if not entries:
            continue

        valid = sorted([e for e in entries if e["t"].size > 0],
                       key=lambda e: pwm_sort_key(e.get("base", "")))
        if not valid:
            continue

        for key, suffix in [("y", "original"), ("y_s", "smoothed")]:
            fig, ax = plt.subplots(figsize=(25, 12), constrained_layout=True)
            x_offset = 0.0
            section_centers = []
            n = len(valid)

            for idx, e in enumerate(valid):
                start_x = x_offset
                t = e["t"] + x_offset
                if t.size == 0:
                    section_centers.append(start_x)
                    continue

                ax.plot(
                    t, e[key],
                    linestyle='-',
                    marker=line_marker,
                    markersize=line_markersize,
                    linewidth=2.0,
                    alpha=1.0,
                    color=pick_color(idx)
                )

                end_x = t[-1]
                section_centers.append((start_x + end_x) / 2.0)

                if idx < n - 1:
                    sep_x = end_x + cont_gap_time_s / 2.0
                    ax.axvline(sep_x, color='k', linewidth=2.0, alpha=1.0)
                    x_offset = end_x + cont_gap_time_s
                else:
                    x_offset = end_x

            ax.set_xlim(0, max(0.1, x_offset))
            ax.margins(x=0)

            # PWM categorical x-axis (truncate safely if fewer files than labels)
            tick_labels = PWM_TICK_LABELS[:len(section_centers)]
            ax.set_xticks(section_centers[:len(tick_labels)])
            ax.set_xticklabels(tick_labels)

            ax.set_xlabel(RMS_XLABEL)
            ax.set_ylabel("RMS (V)")
            ax.grid(alpha=0.3, linewidth=0.4)

            if save_plots:
                fn = os.path.join(out_root, f"motor{i+1}_rms_{suffix}_continuous.pdf")
                fig.savefig(fn, format='pdf', dpi=dpi_out, bbox_inches='tight')
            if show_plots:
                plt.show()
            plt.close(fig)

    # -------- Per motor: PSD continuous (original + smoothed) with PWM categorical x-axis --------
    for i in range(4):
        entries = combined["psd"][i]
        if not entries:
            continue

        valid = sorted([e for e in entries if e["f"].size > 0],
                       key=lambda e: pwm_sort_key(e.get("base", "")))
        if not valid:
            continue

        for key, suffix in [("y", "original"), ("y_s", "smoothed")]:
            fig, ax = plt.subplots(figsize=(25, 12), constrained_layout=True)
            x_offset = 0.0
            section_centers = []
            n = len(valid)

            for idx, e in enumerate(valid):
                start_x = x_offset
                x = e["f"] + x_offset
                if x.size == 0:
                    section_centers.append(start_x)
                    continue

                ax.plot(
                    x, e[key],
                    linestyle='-',
                    marker=line_marker,
                    markersize=line_markersize,
                    linewidth=2.0,
                    alpha=1.0,
                    color=pick_color(idx)
                )

                end_x = x[-1]
                section_centers.append((start_x + end_x) / 2.0)

                if idx < n - 1:
                    sep_x = end_x + cont_gap_freq_hz / 2.0
                    ax.axvline(sep_x, color='k', linewidth=2.0, alpha=1.0)
                    x_offset = end_x + cont_gap_freq_hz
                else:
                    x_offset = end_x

            ax.set_xlim(0, max(0.1, x_offset))
            ax.margins(x=0)

            # PWM categorical x-axis (truncate safely if fewer files than labels)
            tick_labels = PWM_TICK_LABELS[:len(section_centers)]
            ax.set_xticks(section_centers[:len(tick_labels)])
            ax.set_xticklabels(tick_labels)

            ax.set_xlabel(PSD_XLABEL)
            ax.set_ylabel("Power (dB/Hz)")
            ax.grid(alpha=0.3, linewidth=0.4)

            if save_plots:
                fn = os.path.join(out_root, f"motor{i+1}_psd_{suffix}_continuous.pdf")
                fig.savefig(fn, format='pdf', dpi=dpi_out, bbox_inches='tight')
            if show_plots:
                plt.show()
            plt.close(fig)

# ========= FOLDER RUNNER =========
def run_folder(folder_path: str):
    if save_plots:
        ensure_dir(reports_root)

    files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not files:
        print(f"No CSV files in {folder_path}")
        return

    # Pre-scan shortest duration across all files (uniform trimming)
    min_dur_s = prescan_min_duration_s(files)
    if min_dur_s is None:
        print("Could not determine a common minimal duration across files.")
        return
    print(f"[INFO] Uniform plotted duration set to shortest file: {min_dur_s:.3f} s")

    # Process files (populate combined accumulators only)
    for fp in files:
        try:
            process_file_to_combined(fp, uniform_duration_s=min_dur_s)
        except Exception as e:
            print(f"[ERROR] {os.path.basename(fp)}: {e}")

    # Emit combined plots only
    combined_out = os.path.join(reports_root, "_combined_across_files")
    combined_plots_all_files(combined_out)
    print(f"[DONE] Combined plots saved to: {combined_out}")

if __name__ == "__main__":
    run_folder(folder_path)
