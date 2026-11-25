#!/usr/bin/env python3
"""Zero-ML Turntable Video -> Spec-Compliant .VRM (1.0 / 0.x) Studio."""
import argparse
import io
import json
import math
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SZ = 720
N_VIEWS = 40
WORLD_H = 1.58
COARSE_GRID = (56, 82, 44)  # 2x2x2 -> Fine (112, 164, 88)
SUBDIV = 2
TEX_SZ = 1024

BONES = [
    "hips", "spine", "chest", "neck", "head",
    "leftEye", "rightEye",
    "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
    "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
    "rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes",
    "leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes",
]
PARENT = {
    "hips": None, "spine": "hips", "chest": "spine", "neck": "chest", "head": "neck",
    "leftEye": "head", "rightEye": "head",
    "rightShoulder": "chest", "rightUpperArm": "rightShoulder", "rightLowerArm": "rightUpperArm", "rightHand": "rightLowerArm",
    "leftShoulder": "chest", "leftUpperArm": "leftShoulder", "leftLowerArm": "leftUpperArm", "leftHand": "leftLowerArm",
    "rightUpperLeg": "hips", "rightLowerLeg": "rightUpperLeg", "rightFoot": "rightLowerLeg", "rightToes": "rightFoot",
    "leftUpperLeg": "hips", "leftLowerLeg": "leftUpperLeg", "leftFoot": "leftLowerLeg", "leftToes": "leftFoot",
}
BONE_IDX = {b: i for i, b in enumerate(BONES)}

EXPR_NAMES = [
    "aa", "ih", "ou", "ee", "oh",
    "blink", "blinkLeft", "blinkRight",
    "happy", "angry", "sad", "relaxed", "surprised",
]
VRM0_EXPR_MAP = {
    "aa": "a", "ih": "i", "ou": "u", "ee": "e", "oh": "o",
    "blink": "blink", "blinkLeft": "blink_l", "blinkRight": "blink_r",
    "happy": "joy", "angry": "angry", "sad": "sorrow", "relaxed": "fun",
    "surprised": "unknown",
}


# ---------------------------------------------------------------------------
# 1. Video Extraction, Chroma Keying & View Calibration
# ---------------------------------------------------------------------------

def _resolve_binary(name: str, override: str = None) -> str:
    if override:
        p = Path(override)
        if p.is_file():
            return str(p)
        resolved = shutil.which(override)
        if resolved:
            return resolved
        raise FileNotFoundError(f"Executable not found: {override}")
    resolved = shutil.which(name)
    if resolved:
        return resolved
    raise RuntimeError(f"'{name}' was not found on PATH. Please install FFmpeg or pass --ffmpeg.")


def _probe_duration_sec(mp4_path: Path, ffmpeg_bin: str) -> float:
    ffprobe_bin = shutil.which("ffprobe") or str(Path(ffmpeg_bin).with_name("ffprobe"))
    if ffprobe_bin and Path(ffprobe_bin).exists():
        try:
            out = subprocess.check_output(
                [ffprobe_bin, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(mp4_path)],
                stderr=subprocess.DEVNULL, text=True
            ).strip()
            dur = float(out)
            if math.isfinite(dur) and dur > 0.2:
                return min(60.0, max(1.0, dur))
        except Exception:
            pass
    return 10.0


def extract_frames_720(mp4_path: Path, n_views: int = N_VIEWS, ffmpeg_bin: str = None) -> np.ndarray:
    ff = _resolve_binary("ffmpeg", ffmpeg_bin)
    dur = _probe_duration_sec(mp4_path, ff)
    fps = n_views / dur
    cmd = [
        ff, "-v", "error", "-i", str(mp4_path),
        "-vf", f"fps={fps:.5f},crop='min(iw,ih)':'min(iw,ih)',scale={IMG_SZ}:{IMG_SZ}:flags=lanczos",
        "-vframes", str(n_views),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    n_got = len(raw) // (IMG_SZ * IMG_SZ * 3)
    if n_got < 8:
        raise RuntimeError(f"Decoded only {n_got} frames from {mp4_path}; need at least 8 frames.")
    return raw[:n_got * IMG_SZ * IMG_SZ * 3].reshape(n_got, IMG_SZ, IMG_SZ, 3).copy()


def _erode_2d(mask: np.ndarray, k: int = 1) -> np.ndarray:
    cur = mask
    for _ in range(k):
        p = np.pad(cur, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=False)
        cur = (
            p[:, :-2, :-2] & p[:, :-2, 1:-1] & p[:, :-2, 2:]
            & p[:, 1:-1, :-2] & p[:, 1:-1, 1:-1] & p[:, 1:-1, 2:]
            & p[:, 2:, :-2] & p[:, 2:, 1:-1] & p[:, 2:, 2:]
        )
    return cur


def _dilate_2d(mask: np.ndarray, r: int = 1) -> np.ndarray:
    if r <= 0:
        return mask
    p = np.pad(mask, ((0, 0), (0, 0), (r, r)), mode="constant", constant_values=False)
    hx = mask.copy()
    for dx in range(-r, r + 1):
        if dx != 0:
            hx |= p[:, :, r + dx: r + dx + mask.shape[2]]
    py = np.pad(hx, ((0, 0), (r, r), (0, 0)), mode="constant", constant_values=False)
    out = hx.copy()
    for dy in range(-r, r + 1):
        if dy != 0:
            out |= py[:, r + dy: r + dy + mask.shape[1], :]
    return out


def chroma_key_and_despill(frames: np.ndarray, key_th: float = 24.0):
    rgb_f = frames.astype(np.float32)
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    rb_max = np.maximum(r, b)
    gdom = g - rb_max

    is_bg = ((g > 105.0) & (gdom > key_th)) | ((g > 68.0) & (gdom > key_th + 18.0))
    y_floor_band = int(frames.shape[1] * 0.90)
    is_bg[:, y_floor_band:] |= (g[:, y_floor_band:] > rb_max[:, y_floor_band:] + 7.0) & (g[:, y_floor_band:] > 34.0)
    fg = ~is_bg

    p = np.pad(fg.astype(np.uint8), ((0, 0), (1, 1), (1, 1)), mode="constant")
    nbr_sum = (
        p[:, :-2, :-2] + p[:, :-2, 1:-1] + p[:, :-2, 2:]
        + p[:, 1:-1, :-2] + p[:, 1:-1, 2:]
        + p[:, 2:, :-2] + p[:, 2:, 1:-1] + p[:, 2:, 2:]
    )
    fg = (fg & (nbr_sum >= 3)) | ((~fg) & (nbr_sum >= 7))
    fg_color = _erode_2d(fg, k=2)
    fg_inner = _erode_2d(fg, k=4)
    clean = np.clip(rgb_f.copy(), 0.0, 255.0)
    spill = fg & (~fg_inner) & (g > rb_max + 2.0)
    clean[..., 1] = np.where(spill, rb_max + 2.0, g)

    filled = fg_color.copy()
    H_img, W_img = fg.shape[1], fg.shape[2]
    for _ in range(3):
        p_c = np.pad(clean, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
        p_m = np.pad(filled, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=False)
        acc, wsum = np.zeros_like(clean), np.zeros(fg.shape, dtype=np.float32)
        for dy_n in (-1, 0, 1):
            for dx_n in (-1, 0, 1):
                if dy_n == 0 and dx_n == 0:
                    continue
                nm = p_m[:, 1 + dy_n: 1 + dy_n + H_img, 1 + dx_n: 1 + dx_n + W_img]
                acc += p_c[:, 1 + dy_n: 1 + dy_n + H_img, 1 + dx_n: 1 + dx_n + W_img, :] * nm[..., None]
                wsum += nm
        upd = (~filled) & (wsum > 0)
        clean[upd] = acc[upd] / wsum[upd, None]
        filled |= upd

    return fg, fg_color, np.clip(clean, 0.0, 255.0).astype(np.uint8)


def _measure_torso_hw_frame0(fg0: np.ndarray, cx0: float, px_per_m: float) -> float:
    ys, _ = np.nonzero(fg0)
    if len(ys) < 100:
        return 0.130
    y0, y1 = float(ys.min()), float(ys.max())
    ch = max(100.0, y1 - y0)
    hw_list = []
    for y in range(int(y0 + 0.30 * ch), int(y0 + 0.43 * ch)):
        row = fg0[y]
        padded = np.concatenate([[0], row.astype(np.uint8), [0]])
        diff = np.diff(padded.astype(np.int8))
        runs = [(int(s), int(e)) for s, e in zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1) - 1) if s <= cx0 <= e and (e - s) >= 24]
        if runs:
            hw_list.append(0.5 * (runs[0][1] - runs[0][0]) / px_per_m)
    return max(0.110, min(0.185, float(np.median(hw_list)))) if hw_list else 0.130


def calibrate_views(fg: np.ndarray, dir_override: float = 0.0, pose_mode: str = "auto"):
    V, H, W = fg.shape
    cxs_raw = np.zeros(V, dtype=np.float32)
    widths = np.zeros(V, dtype=np.float32)
    y_tops = np.zeros(V, dtype=np.float32)
    y_bots = np.zeros(V, dtype=np.float32)

    for k in range(V):
        ys, xs = np.nonzero(fg[k])
        if len(xs) < 100:
            cxs_raw[k] = W * 0.5; widths[k] = W * 0.25; y_tops[k] = H * 0.08; y_bots[k] = H * 0.94
            continue
        y0, y1 = np.percentile(ys, [0.2, 99.8])
        y_tops[k], y_bots[k] = y0, y1
        bm_sh = (ys >= y0 + 0.15 * (y1 - y0)) & (ys <= y0 + 0.38 * (y1 - y0))
        xs_sh = xs[bm_sh] if bm_sh.sum() >= 30 else xs
        x0, x1 = np.percentile(xs_sh, [1.0, 99.0])
        widths[k] = max(10.0, float(x1 - x0))
        bm = (ys >= y0 + 0.05 * (y1 - y0)) & (ys <= y0 + 0.90 * (y1 - y0))
        cxs_raw[k] = float(xs[bm].mean()) if bm.any() else float(xs.mean())

    p_cx = np.pad(cxs_raw, (2, 2), mode="wrap")
    cxs = (0.10 * p_cx[:-4] + 0.20 * p_cx[1:-3] + 0.40 * p_cx[2:-2] + 0.20 * p_cx[3:-1] + 0.10 * p_cx[4:]).astype(np.float32)
    yb_med, yt_med = float(np.median(y_bots)), float(np.median(y_tops))
    char_h_px = max(220.0, yb_med - yt_med)
    px_per_m = char_h_px / WORLD_H

    pad_w = np.pad(widths, (1, 1), mode="wrap")
    w_sm = 0.25 * pad_w[:-2] + 0.5 * pad_w[1:-1] + 0.25 * pad_w[2:]
    lo1, hi1, lo2, hi2 = int(0.16 * V), int(0.44 * V), int(0.60 * V), int(0.88 * V)
    i1 = lo1 + int(np.argmin(w_sm[lo1:hi1 + 1]))
    i2 = lo2 + int(np.argmin(w_sm[lo2:hi2 + 1]))
    b_lo, b_hi = min(i1 + 3, i2 - 1), max(i1 + 4, i2 - 2)
    i_back = b_lo + int(np.argmax(w_sm[b_lo:b_hi]))

    def _toe_dx(k_idx: int) -> float:
        ys_s, xs_s = np.nonzero(fg[k_idx])
        if len(ys_s) < 100:
            return 0.0
        y0s, y1s = float(y_tops[k_idx]), float(y_bots[k_idx])
        hs = max(40.0, y1s - y0s)
        ank = xs_s[(ys_s >= y0s + 0.84 * hs) & (ys_s <= y0s + 0.92 * hs)]
        toe = xs_s[(ys_s >= y0s + 0.955 * hs) & (ys_s <= y1s)]
        return float(toe.mean() - ank.mean()) if (len(ank) >= 10 and len(toe) >= 5) else 0.0

    turn_dir = (1.0 if (_toe_dx(i1) - _toe_dx(i2)) >= 0.0 else -1.0) if dir_override == 0.0 else float(dir_override)
    w0_ratio = float(w_sm[0] / char_h_px)
    is_tpose = True if pose_mode == "tpose" else (False if pose_mode == "apose" else (w0_ratio > 0.56))
    half_x = max(0.62, 0.5 * (float(w_sm.max()) / char_h_px) * WORLD_H + 0.08)
    half_z = max(0.46, 0.5 * (float(max(w_sm[i1], w_sm[i2])) / char_h_px) * WORLD_H + 0.14)

    w_front, w_back_m, w_side1, w_side2 = max(float(w_sm[0]), 1.0), max(float(w_sm[i_back]), 1.0), float(w_sm[i1]), float(w_sm[i2])

    def quad_angles(k_a, k_b, th_a, th_b, w_wide, w_nar, rising_w):
        ks = np.arange(k_a, k_b + 1, dtype=np.float64)
        u_lin = (ks - k_a) / max(1.0, float(k_b - k_a))
        ws = w_sm[k_a:k_b + 1].astype(np.float64)
        den = max(100.0, w_wide * w_wide - w_nar * w_nar)
        ang = np.arccos(np.sqrt(np.clip((ws * ws - w_nar * w_nar) / den, 0.0, 1.0)))
        raw = (1.0 - ang / (0.5 * math.pi)) if rising_w else (ang / (0.5 * math.pi))
        span = max(1e-4, float(raw[-1] - raw[0]))
        u_mix = np.maximum.accumulate(0.65 * np.clip((raw - raw[0]) / span, 0.0, 1.0) + 0.35 * u_lin)
        u_mix[0] = 0.0; u_mix[-1] = 1.0
        return (th_a + u_mix * (th_b - th_a)) * turn_dir

    r_end = math.sqrt(max(0.0, min(1.0, (w_sm[-1] ** 2 - w_side2 ** 2) / max(100.0, w_front ** 2 - w_side2 ** 2))))
    th_end = max(2.0 * math.pi * (V - 2.5) / V, min(2.0 * math.pi * (V - 0.8) / V, 2.0 * math.pi - math.acos(r_end)))
    thetas = np.zeros(V, dtype=np.float32)
    thetas[0:i1 + 1] = quad_angles(0, i1, 0.0, 0.5 * math.pi, w_front, w_side1, False)
    thetas[i1:i_back + 1] = quad_angles(i1, i_back, 0.5 * math.pi, math.pi, w_back_m, w_side1, True)
    thetas[i_back:i2 + 1] = quad_angles(i_back, i2, math.pi, 1.5 * math.pi, w_back_m, w_side2, False)
    thetas[i2:V] = quad_angles(i2, V - 1, 1.5 * math.pi, th_end, w_front, w_side2, True)

    return {
        "thetas": thetas, "cxs": cxs, "yb_med": yb_med, "yt_med": yt_med, "px_per_m": px_per_m,
        "anchors": (i1, i_back, i2), "is_tpose": is_tpose, "turn_dir": turn_dir,
        "w0_ratio": w0_ratio, "half_x": half_x, "half_z": half_z,
        "torso_hw_est": _measure_torso_hw_frame0(fg[0], float(cxs[0]), px_per_m),
    }


def extract_leg_gap_curve(fg0: np.ndarray, fg_back: np.ndarray, cx0: float, cx_back: float, yb_med: float, px_per_m: float):
    H_img, _ = fg0.shape
    ys_fg, _ = np.nonzero(fg0)
    if len(ys_fg) < 100:
        return None
    y_top, y_bot = float(ys_fg.min()), float(ys_fg.max())
    ch_px = max(100.0, y_bot - y_top)
    rows = []
    for y in range(int(y_top + 0.54 * ch_px), int(y_bot) + 1):
        if y < 0 or y >= H_img:
            continue
        for mask_img, cx, sgn in ((fg0, cx0, 1.0), (fg_back, cx_back, -1.0)):
            padded = np.concatenate([[0], mask_img[y].astype(np.uint8), [0]])
            diff = np.diff(padded.astype(np.int8))
            runs = [(int(s), int(e)) for s, e in zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1) - 1) if (e - s) >= 4]
            l_leg = [r for r in runs if ((0.5 * (r[0] + r[1]) - cx) * sgn) > 0.015 * px_per_m and abs(r[0] - cx) < 0.22 * px_per_m]
            r_leg = [r for r in runs if ((0.5 * (r[0] + r[1]) - cx) * sgn) < -0.015 * px_per_m and abs(r[1] - cx) < 0.22 * px_per_m]
            if not l_leg or not r_leg:
                continue
            lr = min(l_leg, key=lambda r: abs(0.5 * (r[0] + r[1]) - cx))
            rr = min(r_leg, key=lambda r: abs(0.5 * (r[0] + r[1]) - cx))
            gap_px = (lr[0] - rr[1]) if sgn > 0 else (rr[0] - lr[1])
            if gap_px >= 2:
                mid_x_m = ((0.5 * (rr[1] + lr[0]) - cx) * sgn) / px_per_m
                if abs(mid_x_m) <= 0.06:
                    rows.append(((yb_med - float(y)) / px_per_m, mid_x_m))
                    break
    if len(rows) < 4:
        return None
    arr = np.array(rows, dtype=np.float32)
    return arr[np.argsort(arr[:, 0])]


def extract_arm_gap_curves(fg0: np.ndarray, cx0: float, yb_med: float, px_per_m: float):
    H_img, _ = fg0.shape
    ys_fg, _ = np.nonzero(fg0)
    if len(ys_fg) < 100:
        return None
    y_top, y_bot = float(ys_fg.min()), float(ys_fg.max())
    ch_px = max(100.0, y_bot - y_top)
    rows_L, rows_R, torso_w_list = [], [], []
    for y in range(int(y_top + 0.24 * ch_px), int(y_top + 0.62 * ch_px)):
        if y < 0 or y >= H_img:
            continue
        padded = np.concatenate([[0], fg0[y].astype(np.uint8), [0]])
        diff = np.diff(padded.astype(np.int8))
        runs = [(int(s), int(e)) for s, e in zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1) - 1) if (e - s) >= 4]
        t_runs = [r for r in runs if r[0] <= cx0 <= r[1] and (r[1] - r[0]) >= 28]
        if not t_runs:
            continue
        ts, te = t_runs[0]
        wy = (yb_med - float(y)) / px_per_m
        torso_w_list.append(0.5 * (te - ts) / px_per_m)
        r_runs = [r for r in runs if r[1] < ts]
        if r_runs:
            cut_r_m = (0.5 * (r_runs[-1][1] + ts) - cx0) / px_per_m
            if cut_r_m <= -0.10:
                rows_R.append((wy, cut_r_m))
        l_runs = [r for r in runs if r[0] > te]
        if l_runs:
            cut_l_m = (0.5 * (te + l_runs[0][0]) - cx0) / px_per_m
            if cut_l_m >= 0.10:
                rows_L.append((wy, cut_l_m))

    def _contiguous(rows_list):
        if len(rows_list) < 2:
            return rows_list
        kept = [rows_list[0]]
        for r in rows_list[1:]:
            if abs(kept[-1][0] - r[0]) <= 0.030 and abs(kept[-1][1] - r[1]) <= 0.050:
                kept.append(r)
            elif len(kept) < 6:
                kept = [r]
            else:
                break
        return kept

    rows_L, rows_R = _contiguous(rows_L), _contiguous(rows_R)
    return {
        "rows_L": np.array(rows_L, dtype=np.float32) if rows_L else np.zeros((0, 2), dtype=np.float32),
        "rows_R": np.array(rows_R, dtype=np.float32) if rows_R else np.zeros((0, 2), dtype=np.float32),
        "torso_hw": float(np.median(torso_w_list)) if torso_w_list else 0.130,
    }


def _eval_gap_curve(rows: np.ndarray, y_vals: np.ndarray, default_x: float) -> np.ndarray:
    if len(rows) < 2:
        return np.full(len(y_vals), default_x, dtype=np.float32)
    order = np.argsort(rows[:, 0])
    return np.interp(y_vals, rows[order, 0], rows[order, 1], left=rows[order, 1][0], right=rows[order, 1][-1]).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Anatomical Landmark Fitting (Real Chin & Clavicle Detection!)
# ---------------------------------------------------------------------------

def fit_body_params(fg: np.ndarray, clean_rgb: np.ndarray, calib: dict, arm_gaps: dict = None) -> dict:
    px_per_m, yb_med, cxs = calib["px_per_m"], calib["yb_med"], calib["cxs"]
    i1, i_back, i2 = calib["anchors"]
    turn_dir, is_tpose, torso_hw_est = calib["turn_dir"], calib["is_tpose"], calib["torso_hw_est"]
    H = WORLD_H

    def _side_z_samples(y_lo_m: float, y_hi_m: float):
        zs = []
        for k_s, sgn in ((i1, turn_dir), (i2, -turn_dir)):
            ys_s, xs_s = np.nonzero(fg[k_s])
            wy_s = (yb_med - ys_s) / px_per_m
            m = (wy_s >= y_lo_m) & (wy_s <= y_hi_m)
            if m.any():
                zs.append(((xs_s[m] - cxs[k_s]) / px_per_m) * sgn)
        return np.concatenate(zs) if zs else np.array([0.0], dtype=np.float32)

    def _front_span(y_m: float):
        py = int(np.clip(round(yb_med - y_m * px_per_m), 0, IMG_SZ - 1))
        xs = np.flatnonzero(fg[0, py])
        if len(xs) < 4:
            return None
        return ((xs[0] - cxs[0]) / px_per_m, (xs[-1] - cxs[0]) / px_per_m)

    z_face_pts = _side_z_samples(0.87 * H, 0.93 * H)
    z_face_front = max(0.055, min(0.155, float(np.percentile(z_face_pts, 96.0)) if len(z_face_pts) > 5 else 0.105))
    z_torso_pts = _side_z_samples(0.58 * H, 0.68 * H)
    cz_torso = max(-0.025, min(0.055, 0.5 * (float(np.percentile(z_torso_pts, 20.0)) + float(np.percentile(z_torso_pts, 85.0)))))
    z_leg_pts = _side_z_samples(0.14 * H, 0.30 * H)
    cz_legs = max(-0.03, min(0.05, float(np.median(z_leg_pts)) if len(z_leg_pts) > 5 else cz_torso))

    w_hd_list = [0.5 * (sp[1] - sp[0]) for fy in np.linspace(0.88 * H, 0.93 * H, 8) if (sp := _front_span(float(fy))) is not None]
    rx_head = max(0.086, min(0.102, float(np.median(w_hd_list)) if w_hd_list else 0.095))
    rz_head = rx_head * 1.08
    hcx, hcz = 0.0, z_face_front - rz_head
    y_crown = H
    neck_r = max(0.035, min(0.043, 0.40 * rx_head))
    cz_neck = 0.5 * (cz_torso + hcz)

    ys0, xs0 = np.nonzero(fg[0])
    wx0 = (xs0 - cxs[0]) / px_per_m
    wy0 = (yb_med - ys0) / px_per_m

    u_bins = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    r_arm_bins = np.interp(u_bins, [0.0, 0.18, 0.50, 0.85, 1.0], [0.044, 0.036, 0.031, 0.026, 0.023]).astype(np.float32)
    if is_tpose:
        sh_x = max(0.118, min(0.152, torso_hw_est * 1.12))
        m_arm0 = (np.abs(wx0) > torso_hw_est + 0.12) & (wy0 >= 0.65 * H) & (wy0 <= 0.85 * H)
        shoulder_y = max(0.71 * H, min(0.79 * H, float(np.median(wy0[m_arm0])) if m_arm0.sum() >= 20 else 0.753 * H))
        l_pts = wx0[(wx0 > sh_x + 0.10) & (np.abs(wy0 - shoulder_y) < 0.14 * H)]
        r_pts = wx0[(wx0 < -(sh_x + 0.10)) & (np.abs(wy0 - shoulder_y) < 0.14 * H)]
        arm_span = max(0.42, 0.5 * ((float(np.percentile(l_pts, 99.5)) if len(l_pts) >= 10 else 0.62) + abs(float(np.percentile(r_pts, 0.5)) if len(r_pts) >= 10 else -0.62)) - sh_x)
        arm_tip_x = sh_x + arm_span
        hand_len = max(0.125, min(0.155, 0.24 * arm_span))
        wrist_x = arm_tip_x - hand_len

        m_arm_full = (np.abs(wx0) >= sh_x) & (np.abs(wx0) <= wrist_x) & (np.abs(wy0 - shoulder_y) < 0.10 * H)
        if m_arm_full.any():
            ux_pts = (np.abs(wx0[m_arm_full]) - sh_x) / max(0.1, wrist_x - sh_x)
            for bi in range(len(u_bins)):
                bm = np.abs(ux_pts - u_bins[bi]) <= 0.045
                if bm.sum() >= 8:
                    ys_col = wy0[m_arm_full][bm]
                    r_meas = 0.5 * float(ys_col.max() - ys_col.min())
                    r_env = float(np.interp(u_bins[bi], [0.0, 0.20, 0.55, 1.0], [0.046, 0.037, 0.031, 0.025]))
                    r_arm_bins[bi] = max(0.022, min(r_env + (0.015 if u_bins[bi] < 0.65 else 0.006), 0.55 * r_meas + 0.45 * r_env))
        rp = np.pad(r_arm_bins, 1, mode="edge")
        r_arm_bins = (0.25 * rp[:-2] + 0.50 * rp[1:-1] + 0.25 * rp[2:]).astype(np.float32)
        z_arm_pts = _side_z_samples(shoulder_y - 0.04, shoulder_y + 0.04)
        cz_arm = float(np.clip(0.5 * (cz_torso + hcz) * 0.6 + np.median(z_arm_pts) * 0.4, cz_torso - 0.02, cz_torso + 0.04))
        alpha_L = alpha_R = 0.0
    else:
        sh_x = max(0.118, min(0.152, (arm_gaps["torso_hw"] if arm_gaps else torso_hw_est) * 0.98))
        shoulder_y, arm_span, hand_len = 0.76 * H, 0.50, 0.130
        arm_tip_x, wrist_x, cz_arm, alpha_L = sh_x + arm_span, sh_x + arm_span - hand_len, 0.5 * (cz_torso + hcz), math.radians(52.0)
        rows_L = arm_gaps["rows_L"] if arm_gaps is not None else np.zeros((0, 2), dtype=np.float32)
        if len(rows_L) >= 2:
            arm_m = (wx0 > sh_x + 0.05) & (wy0 >= 0.35 * H) & (wy0 <= shoulder_y)
            if arm_m.sum() >= 20:
                d_l = np.hypot(wx0[arm_m] - sh_x, wy0[arm_m] - shoulder_y)
                far = d_l >= np.percentile(d_l, 95)
                alpha_L = max(math.radians(20.0), min(math.radians(74.0), math.atan2(max(0.05, float(shoulder_y - np.median(wy0[arm_m][far]))), max(0.05, float(np.median(wx0[arm_m][far]) - sh_x)))))
                arm_span = max(0.40, float(np.percentile(d_l, 99)))
                arm_tip_x, wrist_x = sh_x + arm_span, sh_x + arm_span - hand_len
        alpha_R = alpha_L

    # Upgrade A: Real Clavicle & Chin landmark detection (prevents giraffe neck!)
    ys_scan = np.linspace(0.85 * H, 0.74 * H, 36, dtype=np.float32)
    ws_scan = np.array([float(sp[1] - sp[0]) if (sp := _front_span(float(ym))) is not None else 0.25 for ym in ys_scan], dtype=np.float32)
    in_nwin = (ys_scan >= 0.785 * H) & (ys_scan <= 0.848 * H)
    idx_nmin = int(np.flatnonzero(in_nwin)[np.argmin(ws_scan[in_nwin])])
    w_neck_min = float(ws_scan[idx_nmin])
    y_clav_raw = float(ys_scan[idx_nmin]) - 0.025
    exp_thresh = max(w_neck_min * 1.24, 0.295)
    for ii in range(idx_nmin + 1, len(ys_scan)):
        if ws_scan[ii] > exp_thresh:
            y_clav_raw = float(ys_scan[ii])
            break
    y_clavicle = float(np.clip(y_clav_raw, shoulder_y + 0.045, 0.815 * H))

    # Side-view Z-step chin detection + front skin lower edge cross-check
    yc_bins = np.linspace(0.77 * H, 0.90 * H, 20, dtype=np.float32)
    zf_bins = np.full(len(yc_bins), hcz + rz_head * 0.5, dtype=np.float32)
    for bi, ym in enumerate(yc_bins):
        zs_b = _side_z_samples(float(ym) - 0.008, float(ym) + 0.008)
        if len(zs_b) >= 4:
            zf_bins[bi] = float(np.percentile(zs_b, 95.0))
    zfp = np.pad(zf_bins, 1, mode="edge")
    zf_sm = 0.25 * zfp[:-2] + 0.50 * zfp[1:-1] + 0.25 * zfp[2:]
    dz_step = zf_sm[2:] - zf_sm[:-2]
    below_nose = yc_bins[1:-1] < 0.865 * H
    y_chin_z = float(yc_bins[1:-1][below_nose][np.argmax(dz_step[below_nose])]) if below_nose.any() else 0.835 * H

    # Front central skin check (~2.5cm strip around cx0)
    cx0_i = int(round(float(cxs[0])))
    skin_ys = []
    for ym in np.linspace(0.79 * H, 0.86 * H, 18):
        py = int(np.clip(round(yb_med - float(ym) * px_per_m), 0, IMG_SZ - 1))
        sl = clean_rgb[0, py, max(0, cx0_i - 8): min(IMG_SZ, cx0_i + 9)].astype(np.float32)
        if len(sl) and float(sl[:, 0].mean()) > float(sl[:, 1].mean()) + 8.0 and float(sl[:, 0].mean()) > float(sl[:, 2].mean()) + 16.0:
            skin_ys.append(float(ym))
    y_chin_est = 0.65 * y_chin_z + 0.35 * (min(skin_ys) + 0.018) if skin_ys else y_chin_z

    y_chin = float(np.clip(y_chin_est, y_clavicle + 0.042, y_clavicle + 0.072))
    head_h = y_crown - y_chin
    y_neck_base = y_clavicle - 0.015

    leg_ys_k = np.array([0.0, 0.055 * H, 0.15 * H, 0.26 * H, 0.36 * H, 0.48 * H], dtype=np.float32)
    leg_cx_L = np.array([0.072, 0.072, 0.072, 0.074, 0.076, 0.076], dtype=np.float32)
    leg_rx_k = np.array([0.044, 0.034, 0.042, 0.038, 0.048, 0.056], dtype=np.float32)
    leg_rz_k = np.array([0.048, 0.036, 0.044, 0.040, 0.050, 0.058], dtype=np.float32)

    leg_gap_rows = extract_leg_gap_curve(fg[0], fg[i_back], float(cxs[0]), float(cxs[i_back]), yb_med, px_per_m)
    for ki in (1, 2, 3):
        ym = float(leg_ys_k[ki])
        py = int(np.clip(round(yb_med - ym * px_per_m), 0, IMG_SZ - 1))
        padded = np.concatenate([[0], fg[0, py].astype(np.uint8), [0]])
        diff = np.diff(padded.astype(np.int8))
        runs = [(int(s), int(e)) for s, e in zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1) - 1) if (e - s) >= 4]
        mid_x = 0.0 if leg_gap_rows is None else float(_eval_gap_curve(leg_gap_rows, np.array([ym], np.float32), 0.0)[0])
        mid_px = cxs[0] + mid_x * px_per_m
        lr = [r for r in runs if 0.5 * (r[0] + r[1]) > mid_px]
        if lr:
            lbest = min(lr, key=lambda r: abs(r[0] - mid_px))
            cx_m = (0.5 * (lbest[0] + lbest[1]) - cxs[0]) / px_per_m
            hw_m = 0.5 * (lbest[1] - lbest[0]) / px_per_m
            if 0.045 <= cx_m <= 0.125:
                leg_cx_L[ki] = cx_m
            leg_rx_k[ki] = max(0.030, min(0.055, hw_m))
            leg_rz_k[ki] = leg_rx_k[ki] * 1.04
    leg_cx_L[0] = leg_cx_L[1]

    z_shoe_pts = _side_z_samples(0.005 * H, 0.055 * H)
    z_shoe_heel = max(cz_legs - 0.075, min(cz_legs - 0.040, float(np.percentile(z_shoe_pts, 5)) if len(z_shoe_pts) > 5 else cz_legs - 0.055))
    z_shoe_toe = max(cz_legs + 0.075, min(cz_legs + 0.125, float(np.percentile(z_shoe_pts, 96)) if len(z_shoe_pts) > 5 else cz_legs + 0.095))
    leg_gap_top = min(0.45 * H, float(leg_gap_rows[:, 0].max())) if (leg_gap_rows is not None and len(leg_gap_rows)) else 0.36 * H

    # Sample bare leg skin RGB from 0.24*H..0.38*H for Color-Aware Skirt Hemline Separation
    leg_skin_samples = []
    y_l0, y_l1 = int(yb_med - 0.38 * H * px_per_m), int(yb_med - 0.24 * H * px_per_m)
    for kk in (0, i_back):
        sub_fg = fg[kk, max(0, y_l0):min(IMG_SZ, y_l1)]
        sub_rgb = clean_rgb[kk, max(0, y_l0):min(IMG_SZ, y_l1)][sub_fg].astype(np.float32)
        if len(sub_rgb):
            warm = (sub_rgb[:, 0] > sub_rgb[:, 2] + 12.0) & (sub_rgb[:, 0] > 110.0) & (sub_rgb[:, 0] > sub_rgb[:, 1] - 5.0)
            if warm.any():
                leg_skin_samples.append(sub_rgb[warm])
    all_ls = np.vstack(leg_skin_samples) if leg_skin_samples else np.zeros((0, 3), dtype=np.float32)
    has_leg_skin = len(all_ls) >= 40
    leg_skin_rgb = np.median(all_ls, axis=0).astype(np.float32) if has_leg_skin else np.array([225.0, 185.0, 165.0], dtype=np.float32)

    return {
        "H": H, "is_tpose": is_tpose, "torso_hw": torso_hw_est, "leg_gap_top": leg_gap_top,
        "hcx": hcx, "hcz": hcz, "rx_head": rx_head, "rz_head": rz_head,
        "y_crown": y_crown, "y_chin": y_chin, "head_h": head_h, "y_clavicle": y_clavicle,
        "neck_r": neck_r, "cz_neck": cz_neck, "y_neck_base": y_neck_base,
        "shoulder_y": shoulder_y, "sh_x": sh_x, "cz_arm": cz_arm,
        "wrist_x": wrist_x, "hand_len": hand_len, "arm_tip_x": arm_tip_x,
        "u_arm_bins": u_bins, "r_arm_bins": r_arm_bins, "alpha_L": alpha_L, "alpha_R": alpha_R,
        "cz_torso": cz_torso, "cz_legs": cz_legs,
        "leg_ys_k": leg_ys_k, "leg_cx_L": leg_cx_L, "leg_rx_k": leg_rx_k, "leg_rz_k": leg_rz_k,
        "z_shoe_heel": z_shoe_heel, "z_shoe_toe": z_shoe_toe,
        "has_leg_skin": has_leg_skin, "leg_skin_rgb": leg_skin_rgb,
    }


def compute_vertex_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    if len(verts) == 0 or len(tris) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    vn = np.zeros_like(verts, dtype=np.float32)
    np.add.at(vn, tris[:, 0], fn); np.add.at(vn, tris[:, 1], fn); np.add.at(vn, tris[:, 2], fn)
    nlen = np.linalg.norm(vn, axis=1, keepdims=True)
    return np.where((nlen < 1e-5).ravel()[:, None], np.array([[0.0, 1.0, 0.0]], dtype=np.float32), vn / np.maximum(nlen, 1e-6)).astype(np.float32)


def _build_loft_shell(rings: list, u_vals: np.ndarray, v_vals: np.ndarray,
                      cap_start: bool = True, cap_end: bool = True,
                      preserve_fold_winding_below_ring: int = -1):
    M, K = len(rings), len(rings[0])
    verts = np.vstack(rings).astype(np.float32)
    uv = np.zeros((M * K, 2), dtype=np.float32)
    for mi in range(M):
        uv[mi * K:(mi + 1) * K, 0] = u_vals
        uv[mi * K:(mi + 1) * K, 1] = v_vals[mi]

    outer_mi = max(0, min(M - 2, preserve_fold_winding_below_ring if preserve_fold_winding_below_ring >= 0 else 0))
    r0_o, r1_o = outer_mi * K, (outer_mi + 1) * K
    c_mid_o = 0.5 * (rings[outer_mi][:-1].mean(axis=0) + rings[outer_mi + 1][:-1].mean(axis=0))
    k_o = K // 4
    a_o, b_o, c_o = r0_o + k_o, r0_o + k_o + 1, r1_o + k_o + 1
    q_mid_o = (verts[a_o] + verts[b_o] + verts[c_o]) / 3.0
    outer_fwd = float(np.dot(np.cross(verts[b_o] - verts[a_o], verts[c_o] - verts[a_o]), q_mid_o - c_mid_o)) >= 0.0

    tris = []
    use_global = (preserve_fold_winding_below_ring >= 0)
    for mi in range(M - 1):
        r0, r1 = mi * K, (mi + 1) * K
        c_mid = 0.5 * (rings[mi][:-1].mean(axis=0) + rings[mi + 1][:-1].mean(axis=0))
        for ki in range(K - 1):
            a, b, c, d = r0 + ki, r0 + ki + 1, r1 + ki + 1, r1 + ki
            if use_global:
                fwd = outer_fwd
            else:
                q_mid = 0.25 * (verts[a] + verts[b] + verts[c] + verts[d])
                fwd = float(np.dot(np.cross(verts[b] - verts[a], verts[c] - verts[a]), q_mid - c_mid)) >= 0.0
            if fwd:
                tris.append((a, b, c)); tris.append((a, c, d))
            else:
                tris.append((a, c, b)); tris.append((a, d, c))

    if cap_start:
        c0, p0_next = verts[:K - 1].mean(axis=0), verts[K:2 * K - 1].mean(axis=0)
        cid = len(verts); verts = np.vstack([verts, c0[None, :].astype(np.float32)])
        uv = np.vstack([uv, np.array([[float(u_vals.mean()), float(v_vals[0])]], dtype=np.float32)])
        out_dir = c0 - p0_next
        for ki in range(K - 1):
            a, b = ki, ki + 1
            n_t = np.cross(verts[b] - verts[cid], verts[a] - verts[cid])
            tris.append((cid, b, a) if float(np.dot(n_t, out_dir)) >= 0.0 else (cid, a, b))

    if cap_end:
        base, prev = (M - 1) * K, (M - 2) * K
        c1, p1_prev = verts[base:base + K - 1].mean(axis=0), verts[prev:prev + K - 1].mean(axis=0)
        cid = len(verts); verts = np.vstack([verts, c1[None, :].astype(np.float32)])
        uv = np.vstack([uv, np.array([[float(u_vals.mean()), float(v_vals[-1])]], dtype=np.float32)])
        out_dir = c1 - p1_prev
        for ki in range(K - 1):
            a, b = base + ki, base + ki + 1
            n_t = np.cross(verts[b] - verts[cid], verts[a] - verts[cid])
            tris.append((cid, a, b) if float(np.dot(n_t, out_dir)) <= 0.0 else (cid, b, a))

    return verts, np.array(tris, dtype=np.int32), uv


def _merge_shells(parts_list: list):
    v_all, t_all, uv_all, tag_all = [], [], [], []
    off = 0
    for verts, tris, uv, tag in parts_list:
        v_all.append(verts.astype(np.float32)); t_all.append((tris + off).astype(np.int32))
        uv_all.append(uv.astype(np.float32))
        tag_all.append(tag if isinstance(tag, np.ndarray) else np.full(len(verts), tag, dtype=np.int32))
        off += len(verts)
    return np.vstack(v_all).astype(np.float32), np.vstack(t_all).astype(np.int32), np.vstack(uv_all).astype(np.float32), np.concatenate(tag_all).astype(np.int32)


# ---------------------------------------------------------------------------
# 3. Procedural Head (Real Neck) + Limbs/Hands + Closed Hip Cap
# ---------------------------------------------------------------------------

def build_procedural_head(params: dict):
    hcx, hcz, rx_h, rz_h = params["hcx"], params["hcz"], params["rx_head"], params["rz_head"]
    y_crown, y_chin, head_h = params["y_crown"], params["y_chin"], params["head_h"]
    neck_r, cz_neck, y_neck_base = params["neck_r"], params["cz_neck"], params["y_neck_base"]

    N_TH = 64
    phis = np.linspace(-math.pi, math.pi, N_TH + 1, dtype=np.float32)
    s_phi = phis / math.pi
    pad_h = 12.0 / TEX_SZ
    u_ring = pad_h + (0.5 + 0.5 * (0.62 * s_phi + 0.38 * np.sin(0.5 * math.pi * s_phi))) * (1.0 - 2.0 * pad_h)

    ys_neck = np.linspace(y_neck_base, y_chin - 0.022, 8, dtype=np.float32)
    ts_head = np.concatenate([np.linspace(0.0, 0.14, 7, endpoint=False), np.linspace(0.14, 0.72, 20, endpoint=False), np.linspace(0.72, 0.992, 12)]).astype(np.float32)
    ys_all = np.concatenate([ys_neck, y_chin + ts_head * head_h])
    y_total = y_crown - y_neck_base

    sin_p, cos_p = np.sin(phis), np.cos(phis)
    front_c = np.maximum(0.0, cos_p)
    rings, v_vals = [], np.zeros(len(ys_all), dtype=np.float32)
    for ri, y in enumerate(ys_all):
        v_vals[ri] = pad_h + np.clip((y_crown - y) / max(1e-4, y_total), 0.0, 1.0) * (1.0 - 2.0 * pad_h)
        if y < y_chin:
            u_n = np.clip((y - y_neck_base) / max(0.02, y_chin - y_neck_base), 0.0, 1.0)
            flare = 1.0 + 0.12 * max(0.0, 1.0 - 2.6 * u_n) ** 2
            rx, rz = neck_r * flare, neck_r * 1.02 * flare
            cz = cz_neck + u_n * (hcz - cz_neck) * 0.45
            rings.append(np.column_stack([hcx + rx * sin_p, np.full_like(phis, y), cz + rz * cos_p]).astype(np.float32))
        else:
            t = float(np.clip((y - y_chin) / head_h, 0.0, 1.0))
            if t < 0.32:
                wt = t / 0.32
                rx = neck_r * 0.92 + (rx_h * 0.96 - neck_r * 0.92) * math.sin(0.5 * math.pi * wt) ** 0.78
                rz_f = neck_r * 0.95 + (rz_h * 0.95 - neck_r * 0.95) * math.sin(0.5 * math.pi * wt) ** 0.65
                rz_b = neck_r * 0.95 + (rz_h * 0.95 - neck_r * 0.95) * math.sin(0.5 * math.pi * wt) ** 0.85
            elif t < 0.66:
                wt = (t - 0.32) / 0.34
                rx, rz_f, rz_b = rx_h * (0.96 + 0.04 * math.sin(math.pi * wt)), rz_h * (0.95 + 0.05 * math.sin(math.pi * wt * 0.8)), rz_h * (0.95 + 0.07 * wt)
            else:
                dome = math.cos(0.5 * math.pi * min(1.0, ((t - 0.66) / 0.34) * 0.985))
                rx, rz_f, rz_b = rx_h * 0.96 * dome, rz_h * 0.96 * dome, rz_h * 1.02 * dome
            jaw_taper = max(0.0, 1.0 - t / 0.34) ** 1.3
            chin_z_boost = 0.006 * math.exp(-((t - 0.06) / 0.08) ** 2) * (front_c ** 2)
            nose = 0.0085 * np.exp(-((phis / 0.16) ** 2) - ((t - 0.42) / 0.10) ** 2)
            cheek = 0.0045 * np.exp(-(((np.abs(phis) - 0.55) / 0.30) ** 2) - ((t - 0.38) / 0.14) ** 2)
            rz_eff = np.where(cos_p >= 0.0, rz_f * (1.0 - 0.12 * jaw_taper), rz_b)
            rx_eff = rx * (1.0 - 0.10 * jaw_taper * (front_c ** 1.5))
            y_chin_dip = 0.005 * math.exp(-((t - 0.02) / 0.05) ** 2) * (front_c ** 2)
            rings.append(np.column_stack([hcx + rx_eff * sin_p, np.full_like(phis, y - y_chin_dip), hcz + rz_eff * cos_p + nose + cheek + chin_z_boost]).astype(np.float32))

    v_main, t_main, uv_main = _build_loft_shell(rings, u_ring, v_vals, cap_start=True, cap_end=True)
    parts = [(v_main, t_main, uv_main, 0)]
    ear_y = y_chin + 0.47 * head_h
    for sgn, u0, u1 in ((1.0, 0.90, 0.985), (-1.0, 0.015, 0.10)):
        ephis = np.linspace(-math.pi, math.pi, 13, dtype=np.float32)
        eu = u0 + ((ephis / math.pi) * 0.5 + 0.5) * (u1 - u0)
        ev = np.linspace(0.88, 0.985, 6, dtype=np.float32)
        e_rings = []
        for ei in range(6):
            s = ei / 5.0
            ss = math.sin(math.pi * (0.15 + 0.7 * s))
            ex_c, ez_c = sgn * (rx_h * 0.88 + 0.008 * math.sin(0.5 * math.pi * s)), hcz - 0.008 - 0.005 * s
            e_rings.append(np.column_stack([
                np.full_like(ephis, ex_c) + sgn * 0.008 * ss * 0.35 * np.sin(ephis),
                ear_y + 0.020 * ss * np.sin(ephis),
                ez_c + 0.011 * ss * np.cos(ephis),
            ]).astype(np.float32))
        ve, te, uve = _build_loft_shell(e_rings, eu, ev, cap_start=True, cap_end=True)
        parts.append((ve, te, uve, 1))

    verts, tris, uv, tags = _merge_shells(parts)
    return {"verts": verts, "tris": tris, "uv": uv, "normals": compute_vertex_normals(verts, tris), "part_tag": tags}


def build_procedural_limbs_and_hands(params: dict):
    """Build Seamless Arm->Wrist->Palm + 5 Fingers + Legs/Shoes + Closed Hip Cap."""
    H, is_tpose, sh_x, shoulder_y, cz_arm = params["H"], params["is_tpose"], params["sh_x"], params["shoulder_y"], params["cz_arm"]
    wrist_x, hand_len = params["wrist_x"], params["hand_len"]
    u_bins, r_arm_bins = params["u_arm_bins"], params["r_arm_bins"]
    leg_ys_k, leg_cx_L, leg_rx_k, leg_rz_k = params["leg_ys_k"], params["leg_cx_L"], params["leg_rx_k"], params["leg_rz_k"]
    cz_legs, cz_torso, z_heel, z_toe = params["cz_legs"], params["cz_torso"], params["z_shoe_heel"], params["z_shoe_toe"]

    parts = []
    def rect_uv(u, v, rect, pad=5.0):
        rx0, ry0, rx1, ry1 = rect
        return (
            ((rx0 + pad + np.clip(u, 0.0, 1.0) * (rx1 - rx0 - 2.0 * pad)) / TEX_SZ).astype(np.float32),
            ((ry0 + pad + np.clip(v, 0.0, 1.0) * (ry1 - ry0 - 2.0 * pad)) / TEX_SZ).astype(np.float32),
        )

    def _pose_arm_pts(pts, side_sgn):
        if is_tpose:
            return pts
        ang = -(params["alpha_L"] if side_sgn > 0 else params["alpha_R"])
        if abs(ang) < 1e-4:
            return pts
        out = pts.copy()
        px_s, py_s = side_sgn * sh_x, shoulder_y
        dx, dy = out[:, 0] - px_s, out[:, 1] - py_s
        rot = ang * side_sgn
        c, s = math.cos(rot), math.sin(rot)
        out[:, 0] = px_s + c * dx - s * dy
        out[:, 1] = py_s + s * dx + c * dy
        return out

    arm_rects = {1: (6.0, 6.0, 342.0, 496.0), -1: (518.0, 6.0, 854.0, 496.0)}
    hand_rects = {1: (348.0, 6.0, 508.0, 496.0), -1: (860.0, 6.0, 1018.0, 496.0)}
    N_ATH = 24
    phis_a = np.linspace(-math.pi, math.pi, N_ATH + 1, dtype=np.float32)
    cp_a, sp_a = np.cos(phis_a), np.sin(phis_a)
    palm_len = 0.46 * hand_len
    us_arm = np.concatenate([
        np.linspace(-0.08, 0.0, 3, endpoint=False), np.linspace(0.0, 0.42, 6, endpoint=False),
        np.linspace(0.42, 0.56, 4, endpoint=False), np.linspace(0.56, 0.96, 6, endpoint=False),
    ]).astype(np.float32)
    tps_palm = np.linspace(0.0, 1.0, 8, dtype=np.float32)

    for sgn in (1.0, -1.0):
        r_wr = float(r_arm_bins[-1])
        rings, u_ring_uv, tag_rings = [], [], []
        for uai in us_arm:
            r = float(np.interp(np.clip(uai, 0.0, 1.0), u_bins, r_arm_bins))
            xx = sgn * (sh_x + uai * (wrist_x - sh_x))
            tw = np.clip((uai - 0.75) / 0.25, 0.0, 1.0)
            rings.append(_pose_arm_pts(np.column_stack([np.full_like(phis_a, xx), shoulder_y + r * (1.0 - 0.22 * tw) * cp_a, cz_arm + r * (1.0 + 0.05 * tw) * sp_a]).astype(np.float32), sgn))
            u_ring_uv.append(0.76 * (uai + 0.08) / 1.08); tag_rings.append(0)
        for tp in tps_palm:
            xx = sgn * (wrist_x + tp * palm_len)
            ry, rz = r_wr * 0.78 * (1.0 - 0.20 * tp), r_wr * (1.05 + 0.42 * math.sin(0.5 * math.pi * tp))
            cy, cz, exp_p = shoulder_y - 0.002 * tp, cz_arm + 0.002 * math.sin(math.pi * tp), 1.0 - 0.22 * tp
            rings.append(_pose_arm_pts(np.column_stack([
                np.full_like(phis_a, xx), cy + ry * np.sign(cp_a) * (np.abs(cp_a) ** exp_p), cz + rz * np.sign(sp_a) * (np.abs(sp_a) ** exp_p),
            ]).astype(np.float32), sgn))
            u_ring_uv.append(0.76 + 0.24 * tp); tag_rings.append(1 if sgn > 0 else 2)

        N_TOT_R = len(rings)
        ua_x, _ = rect_uv(np.array(u_ring_uv, np.float32), np.zeros(N_TOT_R, np.float32), arm_rects[int(sgn)], pad=5.0)
        _, va_y = rect_uv(np.zeros(N_ATH + 1, np.float32), 0.5 + 0.5 * (phis_a / math.pi), arm_rects[int(sgn)], pad=5.0)
        va_arm, ta_arm, _ = _build_loft_shell(rings, np.linspace(0, 1, N_ATH + 1), np.linspace(0, 1, N_TOT_R), cap_start=True, cap_end=True)
        uva_arm = np.zeros((len(va_arm), 2), dtype=np.float32)
        tags_v = np.zeros(len(va_arm), dtype=np.int32)
        for ri in range(N_TOT_R):
            for ki in range(N_ATH + 1):
                vid = ri * (N_ATH + 1) + ki
                uva_arm[vid] = [ua_x[ri], va_y[ki]]; tags_v[vid] = tag_rings[ri]
        uva_arm[N_TOT_R * (N_ATH + 1)] = [ua_x[0], va_y[N_ATH // 2]]
        uva_arm[N_TOT_R * (N_ATH + 1) + 1] = [ua_x[-1], va_y[N_ATH // 2]]
        tags_v[-1] = 1 if sgn > 0 else 2
        parts.append((va_arm, ta_arm, uva_arm, tags_v))

        hx0, hy0, hx1, hy1 = hand_rects[int(sgn)]
        # Tuck thumb root neatly inside palm outline (bx=0.22, bz=+0.019)
        fingers_spec = [
            (0.22, +0.019, +35.0, -10.0, 0.42, 0.0085, 0),
            (0.92, +0.024, +6.5,  -8.0,  0.48, 0.0072, 1),
            (0.96, +0.008, +1.5,  -8.0,  0.54, 0.0074, 2),
            (0.93, -0.009, -2.5,  -9.0,  0.49, 0.0069, 3),
            (0.88, -0.025, -7.5,  -10.0, 0.39, 0.0060, 4),
        ]
        N_FTH, N_FR = 12, 9
        phis_f = np.linspace(-math.pi, math.pi, N_FTH + 1, dtype=np.float32)
        f_strip_h = (hy1 - hy0) / 5.0
        ts_f = np.concatenate([np.linspace(0.0, 0.84, 6, endpoint=False), np.array([0.89, 0.95, 0.992], dtype=np.float32)])
        for bx_f, bz_off, yaw_d, curl_d, l_f, r_b, fidx in fingers_spec:
            sy0, sy1 = hy0 + fidx * f_strip_h, hy0 + (fidx + 1) * f_strip_h - 3.0
            flen, yaw, curl_max = l_f * hand_len, math.radians(yaw_d), math.radians(curl_d)
            root = np.array([wrist_x + bx_f * palm_len, shoulder_y - 0.003, cz_arm + bz_off], dtype=np.float32)
            f_rings = []
            for tf in ts_f:
                ang_c, s_ax = curl_max * (tf ** 1.3), tf * flen
                px_l, pz_l, py_l = root[0] + s_ax * math.cos(yaw), root[2] + s_ax * math.sin(yaw), root[1] + 0.5 * s_ax * math.sin(ang_c)
                rad = r_b * (1.0 - 0.18 * (tf / 0.84)) if tf <= 0.84 else r_b * 0.82 * math.sqrt(max(0.02, 1.0 - ((tf - 0.84) / 0.16) ** 2))
                nx_z, nz_z = -math.sin(yaw), math.cos(yaw)
                f_rings.append(_pose_arm_pts(np.column_stack([sgn * (px_l + rad * nx_z * np.sin(phis_f)), py_l + rad * 0.90 * np.cos(phis_f), pz_l + rad * nz_z * np.sin(phis_f)]).astype(np.float32), sgn))
            vf, tf_tris, _ = _build_loft_shell(f_rings, np.linspace(0, 1, N_FTH + 1), np.linspace(0, 1, N_FR), cap_start=True, cap_end=True)
            uvf = np.zeros((len(vf), 2), dtype=np.float32)
            uf_x, _ = rect_uv(ts_f, np.zeros(N_FR, np.float32), (hx0, sy0, hx1, sy1), pad=3.0)
            _, vf_y = rect_uv(np.zeros(N_FTH + 1, np.float32), 0.5 + 0.5 * (phis_f / math.pi), (hx0, sy0, hx1, sy1), pad=3.0)
            for ri in range(N_FR):
                for ki in range(N_FTH + 1):
                    uvf[ri * (N_FTH + 1) + ki] = [uf_x[ri], vf_y[ki]]
            uvf[N_FR * (N_FTH + 1):] = [float(uf_x.mean()), float(vf_y[N_ATH // 2 if False else N_FTH // 2])]
            parts.append((vf, tf_tris, uvf, 1 if sgn > 0 else 2))

    # Closed Hip Cap (part_tag = 4): closes the mesh between the legs -> (6..506, 502..534)
    pel_rect = (6.0, 502.0, 506.0, 534.0)
    N_PTH, N_PR = 24, 10
    phis_p = np.linspace(-math.pi, math.pi, N_PTH + 1, dtype=np.float32)
    sin_pp, cos_pp = np.sin(phis_p), np.cos(phis_p)
    y_pel_bot = max(0.32 * H, params["leg_gap_top"] - 0.015)
    y_pel_top = 0.55 * H
    ys_pel = np.linspace(y_pel_bot, y_pel_top, N_PR, dtype=np.float32)
    rx_hip = min(0.135, float(leg_cx_L[4]) + float(leg_rx_k[4]) * 0.92)
    rz_hip = min(0.095, float(leg_rz_k[4]) * 1.45)
    p_rings = []
    for ri, yp in enumerate(ys_pel):
        t_p = ri / float(N_PR - 1)
        flare = math.sin(0.5 * math.pi * (0.25 + 0.75 * t_p))
        zc_p = cz_legs + t_p * (cz_torso - cz_legs)
        p_rings.append(np.column_stack([rx_hip * flare * sin_pp, np.full_like(phis_p, yp), zc_p + rz_hip * flare * cos_pp]).astype(np.float32))
    up_v, _ = rect_uv(0.5 + 0.5 * (phis_p / math.pi), np.zeros(N_PTH + 1, np.float32), pel_rect, pad=4.0)
    _, vp_v = rect_uv(np.zeros(N_PR, np.float32), np.linspace(1.0, 0.0, N_PR), pel_rect, pad=4.0)
    vp_v_m, tp_t, uvp_uv = _build_loft_shell(p_rings, up_v, vp_v, cap_start=True, cap_end=True)
    parts.append((vp_v_m, tp_t, uvp_uv, 4))

    # Left & Right Legs + Shoes -> (6..506, 538..1018) and (518..1018, 538..1018)
    leg_rects = {1: (6.0, 538.0, 506.0, 1018.0), -1: (518.0, 538.0, 1018.0, 1018.0)}
    N_LTH, N_LR = 24, 30
    phis_l = np.linspace(-math.pi, math.pi, N_LTH + 1, dtype=np.float32)
    sin_pl, cos_pl = np.sin(phis_l), np.cos(phis_l)
    ys_l = np.concatenate([np.linspace(0.002, 0.065 * H, 8, endpoint=False), np.linspace(0.065 * H, 0.49 * H, N_LR - 8)]).astype(np.float32)
    rx_l_arr, rz_l_arr, cx_l_arr = np.interp(ys_l, leg_ys_k, leg_rx_k), np.interp(ys_l, leg_ys_k, leg_rz_k), np.interp(ys_l, leg_ys_k, leg_cx_L)

    for sgn in (1.0, -1.0):
        ul, _ = rect_uv(0.5 + 0.5 * (phis_l / math.pi), np.zeros(N_LTH + 1, np.float32), leg_rects[int(sgn)], pad=6.0)
        _, vl = rect_uv(np.zeros(N_LR, np.float32), np.linspace(1.0, 0.0, N_LR), leg_rects[int(sgn)], pad=6.0)
        l_rings = []
        for ri, yl in enumerate(ys_l):
            cx_side, rx = sgn * float(cx_l_arr[ri]), float(rx_l_arr[ri])
            if yl < 0.062 * H:
                tf = 1.0 - yl / (0.062 * H)
                boot_w = rx * (1.0 + 0.22 * math.sin(0.5 * math.pi * tf))
                zf, zb = float(rz_l_arr[ri]) + tf * max(0.02, z_toe - cz_legs - float(rz_l_arr[ri])), float(rz_l_arr[ri]) + tf * max(0.01, cz_legs - z_heel - float(rz_l_arr[ri]))
                zc, rz_m = cz_legs + 0.5 * (zf - zb), 0.5 * (zf + zb)
            else:
                boot_w, zc, rz_m = rx, cz_legs, float(rz_l_arr[ri])
            l_rings.append(np.column_stack([cx_side + boot_w * sin_pl, np.full_like(phis_l, yl), zc + rz_m * cos_pl]).astype(np.float32))
        vl_v, tl_t, uvl_uv = _build_loft_shell(l_rings, ul, vl, cap_start=True, cap_end=True)
        parts.append((vl_v, tl_t, uvl_uv, 0))

    verts, tris, uv, tags = _merge_shells(parts)
    return {"verts": verts, "tris": tris, "uv": uv, "normals": compute_vertex_normals(verts, tris), "part_tag": tags}


# ---------------------------------------------------------------------------
# 4. Shared Video Occupancy, Color-Aware Polar-Lofted Torso, & Clean Hair
# ---------------------------------------------------------------------------

def _count_misses(pts_xyz: np.ndarray, fg_masks: np.ndarray, thetas: np.ndarray, cxs: np.ndarray, yb_med: float, px_per_m: float) -> np.ndarray:
    V, H, W = fg_masks.shape
    misses = np.zeros(len(pts_xyz), dtype=np.uint8)
    wx, wy, wz = pts_xyz[:, 0], pts_xyz[:, 1], pts_xyz[:, 2]
    py = np.clip(np.rint(yb_med - wy * px_per_m).astype(np.int32), 0, H - 1)
    for k in range(V):
        ct, st = float(math.cos(thetas[k])), float(math.sin(thetas[k]))
        px = np.rint(cxs[k] + (wx * ct + wz * st) * px_per_m).astype(np.int32)
        misses += (~((px >= 0) & (px < W) & fg_masks[k, py, np.clip(px, 0, W - 1)])).astype(np.uint8)
    return misses


def _dilate_3d(occ: np.ndarray, r: int = 1) -> np.ndarray:
    cur = occ
    for _ in range(r):
        p = np.pad(cur, 1, mode="constant", constant_values=False)
        nxt = cur.copy()
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for oz in (-1, 0, 1):
                    if ox or oy or oz:
                        nxt |= p[1 + ox: 1 + ox + cur.shape[0], 1 + oy: 1 + oy + cur.shape[1], 1 + oz: 1 + oz + cur.shape[2]]
        cur = nxt
    return cur


def _smooth_scalar_field_3d(field: np.ndarray, passes: int = 2) -> np.ndarray:
    cur = field.astype(np.float32)
    for _ in range(passes):
        p0 = np.pad(cur, ((1, 1), (0, 0), (0, 0)), mode="constant"); cur = 0.25 * p0[:-2] + 0.50 * p0[1:-1] + 0.25 * p0[2:]
        p1 = np.pad(cur, ((0, 0), (1, 1), (0, 0)), mode="constant"); cur = 0.25 * p1[:, :-2] + 0.50 * p1[:, 1:-1] + 0.25 * p1[:, 2:]
        p2 = np.pad(cur, ((0, 0), (0, 0), (1, 1)), mode="constant"); cur = 0.25 * p2[:, :, :-2] + 0.50 * p2[:, :, 1:-1] + 0.25 * p2[:, :, 2:]
    return cur


def build_surface_nets_mesh(occ: np.ndarray, dx: float, dy: float, dz: float, half_x: float, half_z: float):
    if not occ.any():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    phi = _smooth_scalar_field_3d(np.pad(occ.astype(np.float32), 1, mode="constant"), passes=2)
    nx, ny, nz = phi.shape
    solid, iso = phi >= 0.5, 0.5
    c000 = solid[:-1, :-1, :-1]
    any_s, all_s = c000.copy(), c000.copy()
    for ox in (0, 1):
        for oy in (0, 1):
            for oz in (0, 1):
                c = solid[ox:nx - 1 + ox, oy:ny - 1 + oy, oz:nz - 1 + oz]
                any_s |= c; all_s &= c
    ax, ay, az = np.nonzero(any_s & (~all_s))
    if not len(ax):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    cell_to_vid = np.full((nx - 1, ny - 1, nz - 1), -1, dtype=np.int32)
    cell_to_vid[ax, ay, az] = np.arange(len(ax), dtype=np.int32)
    pos_sum, edge_cnt = np.zeros((len(ax), 3), dtype=np.float32), np.zeros(len(ax), dtype=np.float32)

    for oy in (0, 1):
        for oz in (0, 1):
            v0, v1 = phi[ax, ay + oy, az + oz], phi[ax + 1, ay + oy, az + oz]
            cr = (v0 >= iso) != (v1 >= iso)
            if cr.any():
                t = np.where(cr, (iso - v0) / np.where(np.abs(v1 - v0) > 1e-6, v1 - v0, 1.0), 0.5)
                pos_sum[cr, 0] += ax[cr] + np.clip(t[cr], 0.0, 1.0); pos_sum[cr, 1] += ay[cr] + oy; pos_sum[cr, 2] += az[cr] + oz; edge_cnt[cr] += 1.0
    for ox in (0, 1):
        for oz in (0, 1):
            v0, v1 = phi[ax + ox, ay, az + oz], phi[ax + ox, ay + 1, az + oz]
            cr = (v0 >= iso) != (v1 >= iso)
            if cr.any():
                t = np.where(cr, (iso - v0) / np.where(np.abs(v1 - v0) > 1e-6, v1 - v0, 1.0), 0.5)
                pos_sum[cr, 0] += ax[cr] + ox; pos_sum[cr, 1] += ay[cr] + np.clip(t[cr], 0.0, 1.0); pos_sum[cr, 2] += az[cr] + oz; edge_cnt[cr] += 1.0
    for ox in (0, 1):
        for oy in (0, 1):
            v0, v1 = phi[ax + ox, ay + oy, az], phi[ax + ox, ay + oy, az + 1]
            cr = (v0 >= iso) != (v1 >= iso)
            if cr.any():
                t = np.where(cr, (iso - v0) / np.where(np.abs(v1 - v0) > 1e-6, v1 - v0, 1.0), 0.5)
                pos_sum[cr, 0] += ax[cr] + ox; pos_sum[cr, 1] += ay[cr] + oy; pos_sum[cr, 2] += az[cr] + np.clip(t[cr], 0.0, 1.0); edge_cnt[cr] += 1.0

    ijk_pos = pos_sum / np.maximum(edge_cnt[:, None], 1.0)
    verts = np.column_stack([-half_x + (ijk_pos[:, 0] - 0.5) * dx, (ijk_pos[:, 1] - 0.5) * dy, -half_z + (ijk_pos[:, 2] - 0.5) * dz]).astype(np.float32)

    tri_chunks = []
    def emit_quads(v00, v10, v11, v01, forward_mask):
        tri_chunks.append(np.where(forward_mask[:, None], np.column_stack([v00, v10, v11]), np.column_stack([v00, v11, v10])))
        tri_chunks.append(np.where(forward_mask[:, None], np.column_stack([v00, v11, v01]), np.column_stack([v00, v01, v11])))

    ex, ey, ez = np.nonzero(solid[:-1, 1:-1, 1:-1] != solid[1:, 1:-1, 1:-1]); ey += 1; ez += 1
    if len(ex): emit_quads(cell_to_vid[ex, ey-1, ez-1], cell_to_vid[ex, ey, ez-1], cell_to_vid[ex, ey, ez], cell_to_vid[ex, ey-1, ez], solid[ex, ey, ez])
    ex, ey, ez = np.nonzero(solid[1:-1, :-1, 1:-1] != solid[1:-1, 1:, 1:-1]); ex += 1; ez += 1
    if len(ex): emit_quads(cell_to_vid[ex-1, ey, ez-1], cell_to_vid[ex-1, ey, ez], cell_to_vid[ex, ey, ez], cell_to_vid[ex, ey, ez-1], solid[ex, ey, ez])
    ex, ey, ez = np.nonzero(solid[1:-1, 1:-1, :-1] != solid[1:-1, 1:-1, 1:]); ex += 1; ey += 1
    if len(ex): emit_quads(cell_to_vid[ex-1, ey-1, ez], cell_to_vid[ex, ey-1, ez], cell_to_vid[ex, ey, ez], cell_to_vid[ex-1, ey, ez], solid[ex, ey, ez])
    return (verts, np.vstack(tri_chunks).astype(np.int32)) if tri_chunks else (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32))


def keep_large_components_frac(verts: np.ndarray, tris: np.ndarray, min_tris: int = 120, min_frac: float = 0.06):
    if len(tris) < 10:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    N = len(verts)
    adj = [[] for _ in range(N)]
    for a, b, c in tris:
        adj[a].extend((b, c)); adj[b].extend((a, c)); adj[c].extend((a, b))
    comp, sizes = np.full(N, -1, dtype=np.int32), []
    for s in range(N):
        if comp[s] >= 0 or not adj[s]:
            continue
        cid, stack, sz = len(sizes), [s], 0; comp[s] = cid
        while stack:
            u = stack.pop(); sz += 1
            for w in adj[u]:
                if comp[w] < 0: comp[w] = cid; stack.append(w)
        sizes.append(sz)
    keep_cids = {i for i, sz in enumerate(sizes) if sz >= max(min_tris // 2, int(N * min_frac))}
    if not keep_cids:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    tris_k = tris[np.array([comp[t[0]] in keep_cids for t in tris], dtype=bool)]
    if len(tris_k) < 16:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    used, inv = np.unique(tris_k.ravel(), return_inverse=True)
    return verts[used].copy(), inv.reshape(-1, 3).astype(np.int32)


def taubin_smooth(verts: np.ndarray, tris: np.ndarray, passes: int = 5, lam: float = 0.46, mu: float = -0.48):
    if len(verts) == 0 or len(tris) == 0:
        return verts
    N = len(verts)
    e0, e1 = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]]), np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    edges = np.unique(np.column_stack([np.minimum(e0, e1), np.maximum(e0, e1)]), axis=0)
    u, v = edges[:, 0], edges[:, 1]
    deg = np.zeros(N, dtype=np.float32); np.add.at(deg, u, 1.0); np.add.at(deg, v, 1.0)
    inv_deg, cur = (1.0 / np.maximum(deg, 1.0))[:, None], verts.astype(np.float32).copy()
    for _ in range(passes):
        for w in (lam, mu):
            nbr_sum = np.zeros_like(cur); np.add.at(nbr_sum, u, cur[v]); np.add.at(nbr_sum, v, cur[u])
            cur += w * (nbr_sum * inv_deg - cur)
    return cur


def _keep_connected_3d_components(seed: np.ndarray, min_vox: int, min_y_idx: int = None) -> np.ndarray:
    if not seed.any():
        return np.zeros_like(seed)
    nx, ny, nz = seed.shape
    vis, out = np.zeros_like(seed), np.zeros_like(seed)
    sx, sy, sz = np.nonzero(seed)
    for start in zip(sx.tolist(), sy.tolist(), sz.tolist()):
        if vis[start]:
            continue
        vis[start] = True; stack, comp, y_hi = [start], [], start[1]
        while stack:
            x, y, z = stack.pop(); comp.append((x, y, z))
            if y > y_hi: y_hi = y
            for ox, oy, oz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                xx, yy, zz = x + ox, y + oy, z + oz
                if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz and seed[xx, yy, zz] and not vis[xx, yy, zz]:
                    vis[xx, yy, zz] = True; stack.append((xx, yy, zz))
        if len(comp) >= min_vox and (min_y_idx is None or y_hi >= min_y_idx):
            pts = np.array(comp, dtype=np.int32)
            out[pts[:, 0], pts[:, 1], pts[:, 2]] = True
    return out


def unwrap_box6_uv(verts: np.ndarray, normals: np.ndarray, tris: np.ndarray):
    if len(verts) == 0 or len(tris) == 0:
        return None
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    dom = np.argmax(np.abs(fn), axis=1)
    cid = np.zeros(len(tris), dtype=np.int32)
    for i in range(len(tris)):
        d = dom[i]
        cid[i] = (0 if fn[i, 2] >= 0.0 else 1) if d == 2 else ((2 if fn[i, 0] >= 0.0 else 3) if d == 0 else (4 if fn[i, 1] >= 0.0 else 5))
    pair = np.column_stack([tris.ravel(), np.repeat(cid, 3)])
    uniq, inv = np.unique(pair, axis=0, return_inverse=True)
    m_tris, orig_v, vcid = inv.reshape(-1, 3).astype(np.int32), uniq[:, 0], uniq[:, 1]
    m_verts, m_norms = verts[orig_v].copy(), normals[orig_v].copy()

    lo, hi = m_verts.min(axis=0), m_verts.max(axis=0)
    xmin, xmax, ymin, ymax, zmin, zmax = float(lo[0])-0.005, float(hi[0])+0.005, float(lo[1])-0.005, float(hi[1])+0.005, float(lo[2])-0.005, float(hi[2])+0.005
    dx, dy, dz = max(0.05, xmax - xmin), max(0.05, ymax - ymin), max(0.05, zmax - zmin)
    rects = {0:(0,0,512,640), 1:(512,0,1024,640), 2:(0,640,256,1024), 3:(256,640,512,1024), 4:(512,640,768,1024), 5:(768,640,1024,1024)}
    m_uv, pad = np.zeros((len(m_verts), 2), dtype=np.float32), 8.0
    vx, vy, vz = m_verts[:, 0], m_verts[:, 1], m_verts[:, 2]
    for k, (rx0, ry0, rx1, ry1) in rects.items():
        mk = vcid == k
        if not mk.any(): continue
        ul, vl = (
            ((vx[mk] - xmin) / dx, (ymax - vy[mk]) / dy) if k == 0 else
            ((xmax - vx[mk]) / dx, (ymax - vy[mk]) / dy) if k == 1 else
            ((zmax - vz[mk]) / dz, (ymax - vy[mk]) / dy) if k == 2 else
            ((vz[mk] - zmin) / dz, (ymax - vy[mk]) / dy) if k == 3 else
            ((vx[mk] - xmin) / dx, (vz[mk] - zmin) / dz) if k == 4 else
            ((vx[mk] - xmin) / dx, (zmax - vz[mk]) / dz)
        )
        m_uv[mk, 0] = (rx0 + pad + np.clip(ul, 0.0, 1.0) * (rx1 - rx0 - 2.0 * pad)) / TEX_SZ
        m_uv[mk, 1] = (ry0 + pad + np.clip(vl, 0.0, 1.0) * (ry1 - ry0 - 2.0 * pad)) / TEX_SZ
    return {"verts": m_verts, "normals": m_norms, "uv": m_uv, "tris": m_tris, "part_tag": np.zeros(len(m_verts), dtype=np.int32)}


def build_video_occupancy(fg: np.ndarray, calib: dict, params: dict):
    thetas, cxs, yb_med, px_per_m = calib["thetas"], calib["cxs"], calib["yb_med"], calib["px_per_m"]
    half_x, half_z, H = calib["half_x"], calib["half_z"], params["H"]
    nx_c, ny_c, nz_c = COARSE_GRID
    nx, ny, nz = nx_c * SUBDIV, ny_c * SUBDIV, nz_c * SUBDIV
    dx_c, dy_c, dz_c = (2.0 * half_x) / nx_c, WORLD_H / ny_c, (2.0 * half_z) / nz_c
    dx, dy, dz = (2.0 * half_x) / nx, WORLD_H / ny, (2.0 * half_z) / nz

    fg_coarse = _dilate_2d(fg, r=int(math.ceil(1.05 * max(dx_c, dy_c) * px_per_m)) + 2)
    fg_hull_body = _dilate_2d(fg, r=2)
    fg_hull_limb = _dilate_2d(fg, r=1)

    gx_c, gy_c, gz_c = np.meshgrid(
        -half_x + (np.arange(nx_c, dtype=np.float32) + 0.5) * dx_c,
        (np.arange(ny_c, dtype=np.float32) + 0.5) * dy_c,
        -half_z + (np.arange(nz_c, dtype=np.float32) + 0.5) * dz_c, indexing="ij"
    )
    occ_c = (_count_misses(np.column_stack([gx_c.ravel(), gy_c.ravel(), gz_c.ravel()]), fg_coarse, thetas, cxs, yb_med, px_per_m) <= 4).reshape(nx_c, ny_c, nz_c)
    p_c = np.pad(occ_c, 1, mode="constant", constant_values=False)
    empty_nbr = np.zeros_like(occ_c)
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            for oz in (-1, 0, 1):
                if ox or oy or oz:
                    empty_nbr |= ~p_c[1 + ox: 1 + ox + nx_c, 1 + oy: 1 + oy + ny_c, 1 + oz: 1 + oz + nz_c]

    occ_video = np.zeros((nx, ny, nz), dtype=bool)
    int_ix, int_iy, int_iz = np.nonzero(occ_c & (~empty_nbr))
    for ox in range(SUBDIV):
        for oy in range(SUBDIV):
            for oz in range(SUBDIV):
                occ_video[int_ix * SUBDIV + ox, int_iy * SUBDIV + oy, int_iz * SUBDIV + oz] = True

    bx, by, bz = np.nonzero(occ_c & empty_nbr)
    offs = np.array([[ox, oy, oz] for ox in range(SUBDIV) for oy in range(SUBDIV) for oz in range(SUBDIV)], dtype=np.int32)
    fx, fy, fz = (bx[:, None] * SUBDIV + offs[None, :, 0]).ravel(), (by[:, None] * SUBDIV + offs[None, :, 1]).ravel(), (bz[:, None] * SUBDIV + offs[None, :, 2]).ravel()
    pts_f = np.column_stack([-half_x + (fx.astype(np.float32) + 0.5) * dx, (fy.astype(np.float32) + 0.5) * dy, -half_z + (fz.astype(np.float32) + 0.5) * dz])
    keep_f = _count_misses(pts_f, fg_hull_body, thetas, cxs, yb_med, px_per_m) <= 3
    occ_video[fx[keep_f], fy[keep_f], fz[keep_f]] = True

    ax_f, ay_f, az_f = np.nonzero(occ_video)
    if len(ax_f):
        pts_act = np.column_stack([-half_x + (ax_f.astype(np.float32) + 0.5) * dx, (ay_f.astype(np.float32) + 0.5) * dy, -half_z + (az_f.astype(np.float32) + 0.5) * dz])
        i1, i_back, i2 = calib["anchors"]
        key_idx = np.array([0, i1, i_back, i2], dtype=np.int32)
        keep_act = _count_misses(pts_act, fg_hull_limb[key_idx], thetas[key_idx], cxs[key_idx], yb_med, px_per_m) == 0
        in_tor_col = (np.abs(pts_act[:, 0]) < params["torso_hw"] + 0.025) & ((pts_act[:, 2] - params["cz_torso"]) > -0.095)
        upper_prot = (pts_act[:, 1] >= params["shoulder_y"] - 0.14 * H) & (~in_tor_col) & (np.abs(pts_act[:, 0]) < params["sh_x"] + 0.16)
        if upper_prot.any():
            keep_act[upper_prot] &= (_count_misses(pts_act[upper_prot], fg_hull_limb, thetas, cxs, yb_med, px_per_m) <= 1)
        drop = ~keep_act
        if drop.any():
            occ_video[ax_f[drop], ay_f[drop], az_f[drop]] = False

    gx = -half_x + (np.arange(nx, dtype=np.float32) + 0.5) * dx
    gy = (np.arange(ny, dtype=np.float32) + 0.5) * dy
    gz = -half_z + (np.arange(nz, dtype=np.float32) + 0.5) * dz
    return occ_video, gx, gy, gz, dx, dy, dz


def build_polar_lofted_torso(occ_video: np.ndarray, gx, gy, gz, dx: float, dy: float, dz: float,
                             fg: np.ndarray, clean_rgb: np.ndarray, calib: dict, params: dict, arm_gaps: dict = None):
    """Build continuous variable-hemline surface with color-aware leg exclusion & seamless clavicle collar."""
    nx, ny, nz = occ_video.shape
    H, is_tpose = params["H"], params["is_tpose"]
    sh_x, shoulder_y, y_clav = params["sh_x"], params["shoulder_y"], params["y_clavicle"]
    y_neck_top = y_clav + 0.002
    neck_r, half_x, half_z = params["neck_r"], calib["half_x"], calib["half_z"]

    VX, VY, VZ = np.meshgrid(gx, gy, gz, indexing="ij")
    occ_src = occ_video.copy()

    # 1. Exclude Head & Protruding Hair
    hair_excl = (VY >= params["y_chin"] - 0.01) | (
        (VY >= shoulder_y - 0.04 * H)
        & (((VZ < params["cz_torso"] - 0.125) & (np.abs(VX) < 0.105)) | ((np.abs(VX) > sh_x * 0.85) & (VY > shoulder_y + 0.025)))
    )
    occ_src &= (~hair_excl)

    # 2. Exclude Arms + Generous Leg Shafts + Color-Aware Bare-Leg Rejection
    if is_tpose:
        y_slope = np.clip((VY - shoulder_y) / max(0.04, y_neck_top - shoulder_y), 0.0, 1.0)
        x_cap = np.where(VY >= shoulder_y, (1.0 - y_slope) * (sh_x + 0.048) + y_slope * (neck_r * 1.45), sh_x + 0.045)
        arm_clip = (np.abs(VX) > x_cap) & (np.abs(VY - shoulder_y) < 0.14 * H)
        occ_src &= (~arm_clip)
    elif arm_gaps is not None:
        rL, rR = arm_gaps["rows_L"], arm_gaps["rows_R"]
        cut_L = _eval_gap_curve(rL, gy, sh_x + 0.03)[None, :, None]
        cut_R = _eval_gap_curve(rR, gy, -(sh_x + 0.03))[None, :, None]
        y_lo_a = max(0.35 * H, (min(float(rL[:, 0].min()) if len(rL) else 0.55, float(rR[:, 0].min()) if len(rR) else 0.55)) - 0.04)
        occ_src &= (~((VY >= y_lo_a) & (VY <= shoulder_y + 0.06) & ((VX > cut_L) | (VX < cut_R))))

    lcx_m = np.interp(gy, params["leg_ys_k"], params["leg_cx_L"])[None, :, None]
    lrx_m = (np.interp(gy, params["leg_ys_k"], params["leg_rx_k"]) * 1.65)[None, :, None]
    lrz_m = (np.interp(gy, params["leg_ys_k"], params["leg_rz_k"]) * 1.65)[None, :, None]
    leg_shaft = (VY < params["leg_gap_top"] + 0.030) & (
        (((VX - lcx_m) / lrx_m) ** 2 + ((VZ - params["cz_legs"]) / lrz_m) ** 2 <= 1.0)
        | (((VX + lcx_m) / lrx_m) ** 2 + ((VZ - params["cz_legs"]) / lrz_m) ** 2 <= 1.0)
    )
    occ_src &= (~leg_shaft)
    occ_src[:, gy < 0.20 * H, :] = False

    if params["has_leg_skin"]:
        thetas, cxs, yb_med, px_per_m = calib["thetas"], calib["cxs"], calib["yb_med"], calib["px_per_m"]
        i_back = calib["anchors"][1]
        ls_rgb = params["leg_skin_rgb"]
        low_cand = occ_src & (VY < params["leg_gap_top"] + 0.045)
        lx, ly, lz = np.nonzero(low_cand)
        if len(lx):
            is_skin_vox = np.ones(len(lx), dtype=bool)
            seen_any = np.zeros(len(lx), dtype=bool)
            for kk in (0, i_back):
                ct, st = float(math.cos(thetas[kk])), float(math.sin(thetas[kk]))
                px_v = np.clip(np.rint(cxs[kk] + (gx[lx] * ct + gz[lz] * st) * px_per_m).astype(np.int32), 0, IMG_SZ - 1)
                py_v = np.clip(np.rint(yb_med - gy[ly] * px_per_m).astype(np.int32), 0, IMG_SZ - 1)
                fg_v = fg[kk, py_v, px_v]
                rgb_v = clean_rgb[kk, py_v, px_v].astype(np.float32)
                near_skin = (np.abs(rgb_v - ls_rgb[None, :]).sum(axis=1) < 115.0) & (rgb_v[:, 0] > rgb_v[:, 2] + 8.0) & (rgb_v[:, 0] > 110.0)
                seen_any |= fg_v
                is_skin_vox &= (~fg_v) | near_skin
            drop_skin = seen_any & is_skin_vox
            if drop_skin.any():
                occ_src[lx[drop_skin], ly[drop_skin], lz[drop_skin]] = False

    # 3. Radial Polar Ray-Cast per (iy, phi)
    N_TH = 64
    phis = np.linspace(-math.pi, math.pi, N_TH, endpoint=False, dtype=np.float32)
    sin_p, cos_p = np.sin(phis), np.cos(phis)
    z_axis_y = np.interp(gy, [0.25 * H, 0.58 * H, y_neck_top], [params["cz_legs"], params["cz_torso"], params["cz_neck"]]).astype(np.float32)

    r_steps = np.linspace(0.36, 0.02, 44, dtype=np.float32)
    px_g = sin_p[None, :, None] * r_steps[None, None, :]
    pz_g = z_axis_y[:, None, None] + cos_p[None, :, None] * r_steps[None, None, :]
    ix_g = np.rint((px_g + half_x) / dx - 0.5).astype(np.int32)
    iz_g = np.rint((pz_g + half_z) / dz - 0.5).astype(np.int32)
    iy_g = np.broadcast_to(np.arange(ny, dtype=np.int32)[:, None, None], iz_g.shape)
    hits = (ix_g >= 0) & (ix_g < nx) & (iz_g >= 0) & (iz_g < nz) & occ_src[np.clip(ix_g, 0, nx - 1), iy_g, np.clip(iz_g, 0, nz - 1)]
    r_raw = np.where(hits.any(axis=2), r_steps[np.argmax(hits, axis=2)], 0.0).astype(np.float32)

    # 4. Variable Hemline + Uniform-Skirt Anti-Spike Clamp
    iy_waist = int(np.clip(round((0.55 * H) / dy), 0, ny - 1))
    iy_min_hem = int(np.clip(round((0.20 * H) / dy), 0, ny - 1))
    iy_gap_max = int(np.clip(round((min(0.47 * H, params["leg_gap_top"] + 0.020)) / dy), iy_min_hem, iy_waist))
    y_hem = np.full(N_TH, float(gy[iy_waist]), dtype=np.float32)
    for k in range(N_TH):
        last_valid, miss_run = iy_waist, 0
        for iy in range(iy_waist, iy_min_hem - 1, -1):
            if r_raw[iy, k] > 0.055:
                last_valid = iy; miss_run = 0
            else:
                miss_run += 1
                if miss_run >= 2: break
        y_hem[k] = float(gy[min(last_valid, iy_gap_max)])

    q25, q50, q75 = np.percentile(y_hem, [25.0, 50.0, 75.0])
    if abs(float(q50) - params["leg_gap_top"]) < 0.080 and float(q75 - q25) < 0.085:
        y_hem = np.clip(y_hem, params["leg_gap_top"] - 0.020, params["leg_gap_top"] + 0.028)
    for _ in range(7):
        y_hem = 0.25 * np.roll(y_hem, 1) + 0.50 * y_hem + 0.25 * np.roll(y_hem, -1)

    # 5. Build (N_RINGS=52, N_TH=64) grid + 2 clean tucked inner skirt lining rings
    N_RINGS = 52
    s_vals, s_waist = np.linspace(0.0, 1.0, N_RINGS, dtype=np.float32), 0.45
    y_grid, r_grid = np.zeros((N_RINGS, N_TH), dtype=np.float32), np.zeros((N_RINGS, N_TH), dtype=np.float32)
    for si, sv in enumerate(s_vals):
        y_row = np.full(N_TH, 0.55 * H + ((sv - s_waist) / (1.0 - s_waist)) * (y_neck_top - 0.55 * H), dtype=np.float32) if sv >= s_waist else y_hem + (sv / s_waist) * (0.55 * H - y_hem)
        y_grid[si] = y_row
        f_iy = np.clip(y_row / dy - 0.5, 0.0, ny - 1.001)
        iy0 = f_iy.astype(np.int32); fy = f_iy - iy0
        r_s = (1.0 - fy) * r_raw[iy0, np.arange(N_TH)] + fy * r_raw[iy0 + 1, np.arange(N_TH)]
        r_grid[si] = np.maximum(r_s, 0.055 if sv < 0.88 else neck_r * 0.90)

    r_grid[-1, :] = neck_r * 0.88
    r_grid[-2, :] = 0.50 * r_grid[-2, :] + 0.50 * (neck_r * 1.02)
    for _ in range(3):
        r_grid = 0.25 * np.roll(r_grid, 1, axis=1) + 0.50 * r_grid + 0.25 * np.roll(r_grid, -1, axis=1)
    for _ in range(3):
        r_new = r_grid.copy(); r_new[1:-1] = 0.25 * r_grid[:-2] + 0.50 * r_grid[1:-1] + 0.25 * r_grid[2:]; r_grid = r_new
    r_grid[-1, :] = neck_r * 0.88

    # 2 clean inner lining rings tucked neatly inside hem (never dangling below y_hem!)
    r_in = np.vstack([r_grid[0] * 0.82, r_grid[0] * 0.95])
    y_in = np.vstack([y_grid[0] + 0.026, y_grid[0] + 0.005])
    r_all, y_all = np.vstack([r_in, r_grid]), np.vstack([y_in, y_grid])
    N_TOT_R = len(r_all)

    r_wrap, y_wrap = np.hstack([r_all, r_all[:, :1]]), np.hstack([y_all, y_all[:, :1]])
    phis_w = np.linspace(-math.pi, math.pi, N_TH + 1, dtype=np.float32)
    sin_pw, cos_pw = np.sin(phis_w), np.cos(phis_w)

    rings, v_vals, tag_rings, pad = [], np.zeros(N_TOT_R, dtype=np.float32), np.zeros(N_TOT_R, dtype=np.int32), 10.0 / TEX_SZ
    for ri in range(N_TOT_R):
        yw = y_wrap[ri]
        zc_w = np.interp(yw, [0.25 * H, 0.58 * H, y_neck_top], [params["cz_legs"], params["cz_torso"], params["cz_neck"]]).astype(np.float32)
        rings.append(np.column_stack([r_wrap[ri] * sin_pw, yw, zc_w + r_wrap[ri] * cos_pw]).astype(np.float32))
        if ri < 2:
            v_vals[ri] = 0.98 - (ri / 2.0) * 0.08
            tag_rings[ri] = 3
        else:
            v_vals[ri] = pad + (1.0 - (ri - 2) / float(N_RINGS - 1)) * (0.90 - pad)

    u_vals = pad + (0.5 + 0.5 * (phis_w / math.pi)) * (1.0 - 2.0 * pad)
    verts, tris, uv = _build_loft_shell(rings, u_vals, v_vals, cap_start=False, cap_end=True, preserve_fold_winding_below_ring=5)
    v_tags = np.repeat(tag_rings, N_TH + 1)
    if len(verts) > len(v_tags):
        v_tags = np.concatenate([v_tags, np.zeros(len(verts) - len(v_tags), dtype=np.int32)])

    zc_grid = np.interp(gy, [0.25 * H, 0.58 * H, y_neck_top], [params["cz_legs"], params["cz_torso"], params["cz_neck"]])[None, :, None]
    rad_xy, phi_vx = np.hypot(VX, VZ - zc_grid), np.arctan2(VX, VZ - zc_grid)
    k_vx = np.clip(np.rint((phi_vx + math.pi) / (2.0 * math.pi) * N_TH).astype(np.int32), 0, N_TH - 1)
    r_surf_vx = np.zeros_like(rad_xy)
    for k in range(N_TH):
        mk = (k_vx == k)
        r_surf_vx[mk] = np.interp(VY[mk], y_grid[:, k], r_grid[:, k], left=0.0, right=neck_r * 1.0)
    occ_torso_fit = (VY >= y_hem[k_vx] - 0.01) & (VY <= y_neck_top + 0.01) & (rad_xy <= r_surf_vx + 0.015)

    return {"verts": verts, "tris": tris, "uv": uv, "normals": compute_vertex_normals(verts, tris), "part_tag": v_tags, "occ_torso_fit": occ_torso_fit, "y_hem": y_hem}


def carve_color_consistent_hair(occ_video: np.ndarray, occ_torso_fit: np.ndarray, gx, gy, gz,
                                  dx: float, dy: float, dz: float, fg: np.ndarray, clean_rgb: np.ndarray,
                                  calib: dict, params: dict):
    nx, ny, nz = occ_video.shape
    H, is_tpose, half_x, half_z = params["H"], params["is_tpose"], calib["half_x"], calib["half_z"]
    VX, VY, VZ = np.meshgrid(gx, gy, gz, indexing="ij")

    head_in = (
        ((VX - params["hcx"]) / (params["rx_head"] * 1.04)) ** 2
        + ((VY - (params["y_chin"] + 0.50 * params["head_h"])) / (0.52 * params["head_h"])) ** 2
        + ((VZ - params["hcz"]) / (params["rz_head"] * 1.04)) ** 2 <= 1.08
    )
    neck_in = (VY >= params["y_neck_base"]) & (VY < params["y_chin"]) & (((VX - params["hcx"]) ** 2 + (VZ - params["cz_neck"]) ** 2) <= (params["neck_r"] * 1.15) ** 2)
    arm_in = np.zeros((nx, ny, nz), dtype=bool)
    arm_excl = np.zeros((nx, ny, nz), dtype=bool)
    if is_tpose:
        arm_in = (np.abs(VX) >= params["sh_x"]) & (np.abs(VX) <= params["arm_tip_x"] + 0.01) & (np.abs(VY - params["shoulder_y"]) < 0.06) & (np.abs(VZ - params["cz_arm"]) < 0.06)
        arm_excl = (np.abs(VX) > params["sh_x"] + 0.015) & (np.abs(VY - params["shoulder_y"]) < 0.14 * H) & (VY < params["y_chin"] - 0.01)

    occ_core_dil = _dilate_3d(head_in | neck_in | occ_torso_fit | arm_in, r=1)
    face_clear = (VZ > params["hcz"] + 0.008) & (np.abs(VX - params["hcx"]) < params["rx_head"] * 0.92) & (VY >= params["y_neck_base"] - 0.02) & (VY <= params["y_crown"] - 0.020)

    i_back = calib["anchors"][1]
    thetas, cxs, yb_med, px_per_m = calib["thetas"], calib["cxs"], calib["yb_med"], calib["px_per_m"]
    y_h0, y_h1 = int(np.clip(round(yb_med - (params["y_crown"] - 0.03) * px_per_m), 0, IMG_SZ - 1)), int(np.clip(round(yb_med - (params["y_chin"] + 0.04) * px_per_m), 0, IMG_SZ - 1))
    hair_bg_px = clean_rgb[i_back, min(y_h0, y_h1):max(y_h0, y_h1) + 1][fg[i_back, min(y_h0, y_h1):max(y_h0, y_h1) + 1]]
    is_dark_hair = (len(hair_bg_px) > 20) and (float(np.median(hair_bg_px)) < 125.0)

    hair_zone = (VY >= params["y_chin"] - 0.005) | (
        (VY >= params["shoulder_y"] - 0.05 * H)
        & (((VZ < params["cz_torso"] - 0.130) & (np.abs(VX) < 0.105)) | ((np.abs(VX) > params["sh_x"] * 0.85) & (VY > params["shoulder_y"] + 0.025)))
    )
    hair_cand = occ_video & (~occ_core_dil) & hair_zone & (~face_clear) & (~arm_excl)

    if is_dark_hair:
        lx, ly, lz = np.nonzero(hair_cand & (VY < params["y_chin"] - 0.005))
        if len(lx):
            ct, st = float(math.cos(thetas[i_back])), float(math.sin(thetas[i_back]))
            px_b = np.clip(np.rint(cxs[i_back] + (gx[lx] * ct + gz[lz] * st) * px_per_m).astype(np.int32), 0, IMG_SZ - 1)
            py_b = np.clip(np.rint(yb_med - gy[ly] * px_per_m).astype(np.int32), 0, IMG_SZ - 1)
            col_b = clean_rgb[i_back, py_b, px_b].astype(np.float32)
            bad_col = (col_b.min(axis=1) > 145.0) | (col_b.mean(axis=1) > 165.0)
            if bad_col.any():
                hair_cand[lx[bad_col], ly[bad_col], lz[bad_col]] = False

    hair_seed = _keep_connected_3d_components(hair_cand, min_vox=120, min_y_idx=int((params["y_chin"] - 0.01) / dy))
    if not hair_seed.any():
        return None
    occ_hair = _dilate_3d(hair_seed, r=2) & occ_video & hair_zone & (~arm_excl) & (~face_clear)
    occ_hair |= _dilate_3d(occ_hair, r=1) & occ_video & (~face_clear) & (VY >= params["y_chin"] - 0.005)
    vm, tm = build_surface_nets_mesh(occ_hair, dx, dy, dz, half_x, half_z)
    vm, tm = keep_large_components_frac(vm, tm, min_tris=120, min_frac=0.06)
    if len(tm) < 60:
        return None
    vm = taubin_smooth(vm, tm, passes=5)
    return unwrap_box6_uv(vm, compute_vertex_normals(vm, tm), tm)


# ---------------------------------------------------------------------------
# 5. Combined Scene Z-Buffer Baking with Directional Shoulder & Underside Guards
# ---------------------------------------------------------------------------

def _despeckle_uv_outliers(tex_img: np.ndarray, tex_cov: np.ndarray, diff_thresh: float = 80.0) -> np.ndarray:
    p_cov, p_rgb = np.pad(tex_cov, 1, mode="constant", constant_values=False), np.pad(tex_img.astype(np.int16), ((1, 1), (1, 1), (0, 0)), mode="edge")
    cy, cx = np.nonzero(tex_cov)
    if not len(cy):
        return tex_img
    offs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    nbr_vals, nbr_ok = np.empty((8, len(cy), 3), dtype=np.int16), np.empty((8, len(cy)), dtype=bool)
    for k, (dy, dx) in enumerate(offs):
        nbr_ok[k], nbr_vals[k] = p_cov[cy + 1 + dy, cx + 1 + dx], p_rgb[cy + 1 + dy, cx + 1 + dx]
    cnt = nbr_ok.sum(axis=0); strong = cnt >= 5
    if not strong.any():
        return tex_img
    sy, sx = cy[strong], cx[strong]
    sentinel = np.where(nbr_ok[:, strong, None], nbr_vals[:, strong, :], 999); sentinel.sort(axis=0)
    med = sentinel[np.clip((cnt[strong] - 1) // 2, 0, 7), np.arange(len(sy)), :]
    bad = np.abs(tex_img[sy, sx].astype(np.int16) - med).sum(axis=1) > diff_thresh
    if bad.any():
        out = tex_img.copy(); out[sy[bad], sx[bad]] = np.clip(med[bad], 0, 255).astype(np.uint8); return out
    return tex_img


def _dilate_uv_gutter(tex_img: np.ndarray, tex_cov: np.ndarray, passes: int = 8) -> np.ndarray:
    out, valid = tex_img.copy(), tex_cov.copy()
    for _ in range(passes):
        cur_m = valid.copy()
        for dy_g, dx_g in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            s_m, s_c = np.roll(cur_m, (dy_g, dx_g), axis=(0, 1)), np.roll(out, (dy_g, dx_g), axis=(0, 1))
            take = (~valid) & s_m
            out[take] = s_c[take]; valid |= take
    return out


def _inpaint_vertical_down(tex_img: np.ndarray, valid_mask: np.ndarray, need_mask: np.ndarray, max_r: int = 64):
    """Fill shoulder/collar texels strictly downward in UV (= from blouse/sleeve directly below each longitude)."""
    ys, xs = np.nonzero(need_mask)
    H_t = tex_img.shape[0]
    for y, x in zip(ys, xs):
        found = False
        for d in range(1, max_r + 1):
            if y + d < H_t and valid_mask[y + d, x]:
                tex_img[y, x] = tex_img[y + d, x]
                valid_mask[y, x] = True; need_mask[y, x] = False; found = True; break
        if not found:
            for d in range(1, max_r + 1):
                if y - d >= 0 and valid_mask[y - d, x]:
                    tex_img[y, x] = tex_img[y - d, x]
                    valid_mask[y, x] = True; need_mask[y, x] = False; break


def _inpaint_horizontal_gaps(tex_img: np.ndarray, valid_mask: np.ndarray, need_mask: np.ndarray, max_r: int = 64):
    ys, xs = np.nonzero(need_mask)
    W_t = tex_img.shape[1]
    for y, x in zip(ys, xs):
        lx, rx = -1, -1
        for d in range(1, max_r + 1):
            if lx < 0 and x - d >= 0 and valid_mask[y, x - d]: lx = x - d
            if rx < 0 and x + d < W_t and valid_mask[y, x + d]: rx = x + d
            if lx >= 0 and rx >= 0: break
        if lx >= 0 and rx >= 0:
            t = (x - lx) / float(rx - lx)
            tex_img[y, x] = np.clip((1.0 - t) * tex_img[y, lx].astype(np.float32) + t * tex_img[y, rx].astype(np.float32), 0, 255).astype(np.uint8)
        elif lx >= 0: tex_img[y, x] = tex_img[y, lx]
        elif rx >= 0: tex_img[y, x] = tex_img[y, rx]


def bake_textures(parts: dict, fg: np.ndarray, fg_color: np.ndarray, clean_rgb: np.ndarray, calib: dict, params: dict):
    thetas, cxs, yb_med, px_per_m = calib["thetas"], calib["cxs"], calib["yb_med"], calib["px_per_m"]
    V, ZBUF = len(thetas), 360
    z_scale = ZBUF / float(IMG_SZ)
    front_z = np.full((V, ZBUF, ZBUF), -1e9, dtype=np.float32)

    pt_chunks = []
    for pname in ("head", "torso", "body", "hair"):
        part = parts.get(pname)
        if part is None or len(part["tris"]) == 0:
            continue
        vv, tt = part["verts"], part["tris"]
        v0, v1, v2 = vv[tt[:, 0]], vv[tt[:, 1]], vv[tt[:, 2]]
        pt_chunks.extend([vv, (v0 + v1 + v2) / 3.0, 0.5 * (v0 + v1), 0.5 * (v1 + v2), 0.5 * (v2 + v0)])
    z_pts = np.vstack(pt_chunks).astype(np.float32)

    for k in range(V):
        ct, st = float(math.cos(thetas[k])), float(math.sin(thetas[k]))
        u_w, z_cam = z_pts[:, 0] * ct + z_pts[:, 2] * st, -z_pts[:, 0] * st + z_pts[:, 2] * ct
        px_b = np.clip(np.rint((cxs[k] + u_w * px_per_m) * z_scale).astype(np.int32), 0, ZBUF - 1)
        py_b = np.clip(np.rint((yb_med - z_pts[:, 1] * px_per_m) * z_scale).astype(np.int32), 0, ZBUF - 1)
        np.maximum.at(front_z[k], (py_b, px_b), z_cam)
        pz = np.pad(front_z[k], 1, mode="edge")
        pooled = pz[1:-1, 1:-1].copy()
        for dy_b in (-1, 0, 1):
            for dx_b in (-1, 0, 1):
                pooled = np.maximum(pooled, pz[1 + dy_b: 1 + dy_b + ZBUF, 1 + dx_b: 1 + dx_b + ZBUF])
        front_z[k] = pooled

    clean_f = clean_rgb.astype(np.float32)
    target_angs = [0.0, 0.25 * math.pi, 0.5 * math.pi, 0.75 * math.pi, math.pi, 1.25 * math.pi, 1.5 * math.pi, 1.75 * math.pi]
    ang_mod = np.mod(thetas, 2.0 * math.pi)
    canon_ks = {int(np.argmin(np.minimum(np.abs(ang_mod - ta), 2.0 * math.pi - np.abs(ang_mod - ta)))) for ta in target_angs}
    skirt_hem_rgb = np.array([38.0, 42.0, 56.0], dtype=np.float32)

    for pname in ("head", "torso", "body", "hair"):
        part = parts.get(pname)
        if part is None or len(part["tris"]) == 0:
            continue
        m_verts, m_norms, m_uv, m_tris, m_tags = part["verts"], part["normals"], part["uv"], part["tris"], part["part_tag"]
        uv_px = m_uv * float(TEX_SZ)
        tex_pos, tex_nrm = np.zeros((TEX_SZ, TEX_SZ, 3), dtype=np.float32), np.zeros((TEX_SZ, TEX_SZ, 3), dtype=np.float32)
        tex_cov, tex_tag = np.zeros((TEX_SZ, TEX_SZ), dtype=bool), np.zeros((TEX_SZ, TEX_SZ), dtype=np.int32)

        u0v, v0v = uv_px[m_tris[:, 0], 0], uv_px[m_tris[:, 0], 1]
        u1v, v1v = uv_px[m_tris[:, 1], 0], uv_px[m_tris[:, 1], 1]
        u2v, v2v = uv_px[m_tris[:, 2], 0], uv_px[m_tris[:, 2], 1]
        x0b, y0b = np.clip(np.floor(np.minimum(np.minimum(u0v, u1v), u2v)).astype(np.int32), 0, TEX_SZ - 1), np.clip(np.floor(np.minimum(np.minimum(v0v, v1v), v2v)).astype(np.int32), 0, TEX_SZ - 1)
        x1b, y1b = np.clip(np.ceil(np.maximum(np.maximum(u0v, u1v), u2v)).astype(np.int32), 0, TEX_SZ - 1), np.clip(np.ceil(np.maximum(np.maximum(v0v, v1v), v2v)).astype(np.int32), 0, TEX_SZ - 1)
        p0t, p1t, p2t = m_verts[m_tris[:, 0]], m_verts[m_tris[:, 1]], m_verts[m_tris[:, 2]]
        n0t, n1t, n2t = m_norms[m_tris[:, 0]], m_norms[m_tris[:, 1]], m_norms[m_tris[:, 2]]
        t_tags = np.maximum(np.maximum(m_tags[m_tris[:, 0]], m_tags[m_tris[:, 1]]), m_tags[m_tris[:, 2]])

        area = (u1v - u0v) * (v2v - v0v) - (u2v - u0v) * (v1v - v0v)
        for ti in np.flatnonzero(np.abs(area) > 1e-5):
            ax0, ay0, ax1, ay1 = int(x0b[ti]), int(y0b[ti]), int(x1b[ti]), int(y1b[ti])
            px_g, py_g = np.meshgrid(np.arange(ax0, ax1 + 1, dtype=np.float32) + 0.5, np.arange(ay0, ay1 + 1, dtype=np.float32) + 0.5)
            invA = 1.0 / float(area[ti])
            w0 = ((u1v[ti] - px_g) * (v2v[ti] - py_g) - (u2v[ti] - px_g) * (v1v[ti] - py_g)) * invA
            w1 = ((u2v[ti] - px_g) * (v0v[ti] - py_g) - (u0v[ti] - px_g) * (v2v[ti] - py_g)) * invA
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -0.02) & (w1 >= -0.02) & (w2 >= -0.02)
            if not inside.any(): continue
            iy_s, ix_s = np.nonzero(inside); ty, tx = ay0 + iy_s, ax0 + ix_s
            b0, b1, b2 = np.clip(w0[iy_s, ix_s], 0.0, 1.0)[:, None], np.clip(w1[iy_s, ix_s], 0.0, 1.0)[:, None], np.clip(w2[iy_s, ix_s], 0.0, 1.0)[:, None]
            bs = np.maximum(b0 + b1 + b2, 1e-6); b0 /= bs; b1 /= bs; b2 /= bs
            tex_pos[ty, tx] = b0 * p0t[ti] + b1 * p1t[ti] + b2 * p2t[ti]
            tn = b0 * n0t[ti] + b1 * n1t[ti] + b2 * n2t[ti]
            tex_nrm[ty, tx] = tn / np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-6)
            tex_cov[ty, tx] = True; tex_tag[ty, tx] = int(t_tags[ti])

        cov_ys, cov_xs = np.nonzero(tex_cov)
        Nt = len(cov_ys)
        if Nt == 0:
            part["tex"] = np.full((TEX_SZ, TEX_SZ, 3), 140, dtype=np.uint8); continue

        q_pos, q_nrm, q_tag = tex_pos[cov_ys, cov_xs], tex_nrm[cov_ys, cov_xs], tex_tag[cov_ys, cov_xs]
        is_head, is_body, is_torso, is_hair = (pname == "head"), (pname == "body"), (pname == "torso"), (pname == "hair")
        cz_ref = params["hcz"] if is_head else params["cz_torso"]
        nh_raw = np.hypot(q_nrm[:, 0], q_nrm[:, 2])
        use_fwd = nh_raw < 0.14
        nh_x = np.where(use_fwd, 0.0, q_nrm[:, 0] / np.maximum(nh_raw, 1e-4))
        nh_z = np.where(use_fwd, np.clip((q_pos[:, 2] - cz_ref) / 0.025, -1.0, 1.0), q_nrm[:, 2] / np.maximum(nh_raw, 1e-4))
        front_dome = np.clip((nh_z - 0.15) / 0.55, 0.0, 1.0)

        scores_all, rgb_all = np.full((V, Nt), -1.0, dtype=np.float32), np.zeros((V, Nt, 3), dtype=np.float32)
        fb_w, fb_rgb = np.zeros(Nt, dtype=np.float32), np.zeros((Nt, 3), dtype=np.float32)
        unocc_tol = -0.024 if is_head else (-0.012 if is_hair else -0.015)

        for k in range(V):
            ct, st = float(math.cos(thetas[k])), float(math.sin(thetas[k]))
            facing_raw = -nh_x * st + nh_z * ct
            facing_flat = 0.55 * abs(ct) + 0.45 * np.where(q_pos[:, 2] >= cz_ref, max(ct, 0.0), max(-ct, 0.0))
            facing = np.where(use_fwd, facing_flat, facing_raw)
            u_w, z_cam = q_pos[:, 0] * ct + q_pos[:, 2] * st, -q_pos[:, 0] * st + q_pos[:, 2] * ct
            px_f, py_f = cxs[k] + u_w * px_per_m, yb_med - q_pos[:, 1] * px_per_m
            px0, py0 = np.floor(px_f).astype(np.int32), np.floor(py_f).astype(np.int32)
            in_img = (px0 >= 0) & (px0 < IMG_SZ - 1) & (py0 >= 0) & (py0 < IMG_SZ - 1)
            px0_c, py0_c = np.clip(px0, 0, IMG_SZ - 2), np.clip(py0, 0, IMG_SZ - 2)
            fx_f, fy_f = np.clip(px_f - px0_c, 0.0, 1.0)[:, None], np.clip(py_f - py0_c, 0.0, 1.0)[:, None]
            px_n, py_n = np.clip(np.rint(px_f).astype(np.int32), 0, IMG_SZ - 1), np.clip(np.rint(py_f).astype(np.int32), 0, IMG_SZ - 1)
            color_ok, fg_ok = in_img & fg_color[k, py_n, px_n], in_img & fg[k, py_n, px_n]
            px_b, py_b = np.clip(np.rint(px_f * z_scale).astype(np.int32), 0, ZBUF - 1), np.clip(np.rint(py_f * z_scale).astype(np.int32), 0, ZBUF - 1)
            unocc = z_cam >= (front_z[k, py_b, px_b] + unocc_tol)

            img_k = clean_f[k]
            rgb_k = (1.0 - fy_f) * ((1.0 - fx_f) * img_k[py0_c, px0_c] + fx_f * img_k[py0_c, px0_c + 1]) + fy_f * ((1.0 - fx_f) * img_k[py0_c + 1, px0_c] + fx_f * img_k[py0_c + 1, px0_c + 1])
            rgb_all[k] = rgb_k
            w_fb = np.where(fg_ok, np.maximum(facing, 0.04) * np.where(unocc, 1.0, 0.15), 0.0)
            fb_w += w_fb; fb_rgb += rgb_k * w_fb[:, None]

            canon_bonus = 0.07 if k in canon_ks else 0.0
            if is_head:
                near_k0 = max(0.0, 1.0 - abs(float(math.atan2(st, ct))) / 0.40)
                tail_pen = np.where(nh_z > 0.15, 0.75 * max(0.0, min(1.0, (k - 0.58 * V) / max(1.0, 0.25 * V))), 0.0)
                scores_all[k] = np.where(color_ok & unocc & (facing > 0.28), facing + canon_bonus + 0.52 * front_dome * near_k0 - front_dome * tail_pen, -1.0)
            else:
                tail_pen = 0.22 * max(0.0, (k - (V - 6)) / 5.0)
                scores_all[k] = np.where(color_ok & unocc & (facing > 0.30), facing + canon_bonus - front_dome * tail_pen, -1.0)

        best_score = scores_all.max(axis=0)
        has_wta = best_score > 0.0
        if is_torso:
            guard_up = (q_nrm[:, 1] > 0.62) | (q_tag == 3)
            has_wta &= (~guard_up)
        if is_hair:
            has_wta &= (q_nrm[:, 1] >= -0.45)

        cone_w_arr = np.where(front_dome > 0.35, 0.055, 0.12)[None, :] if is_head else 0.065
        sig_w_arr = np.where(front_dome > 0.35, 0.024, 0.048)[None, :] if is_head else 0.028
        in_cone = (scores_all >= (best_score[None, :] - cone_w_arr)) & (scores_all > 0.0)
        diff = np.maximum(0.0, best_score[None, :] - scores_all)
        w_cone = np.where(in_cone, np.exp(-((diff / sig_w_arr) ** 2)), 0.0)
        cone_rgb = np.einsum("vn,vnc->nc", w_cone, rgb_all) / np.maximum(w_cone.sum(axis=0), 1e-6)[:, None]

        fb_def = np.median(cone_rgb[has_wta], axis=0)[None, :] if (is_hair and has_wta.any()) else 140.0
        fb_res = np.where((fb_w > 0.0)[:, None], fb_rgb / np.maximum(fb_w[:, None], 1e-6), fb_def)
        sampled = np.clip(np.where(has_wta[:, None], cone_rgb, fb_res), 0.0, 255.0)

        if is_torso:
            hem_ref = has_wta & (q_pos[:, 1] < 0.48 * params["H"])
            if hem_ref.any():
                skirt_hem_rgb = np.median(sampled[hem_ref], axis=0).astype(np.float32)

        if is_hair and has_wta.any():
            hair_all_med = np.median(sampled[has_wta], axis=0).astype(np.float32)
            up_hair = has_wta & (q_pos[:, 1] > params["y_chin"] + 0.01)
            if up_hair.sum() >= 20:
                up_med = np.median(sampled[up_hair], axis=0)
                if up_med.mean() < 125.0:
                    white_bleed = (q_pos[:, 1] <= params["y_chin"] + 0.01) & ((sampled.min(axis=1) > 135.0) | (sampled.mean(axis=1) > 155.0))
                    if white_bleed.any():
                        sampled[white_bleed] = up_med[None, :]
            # Downward-facing hair underside: dark shaded hair tone + subtle strand variation (no grey cap!)
            under_hair = (q_nrm[:, 1] < -0.45)
            if under_hair.any():
                strand = (0.94 + 0.08 * np.sin(cov_xs[under_hair] * 0.45 + cov_ys[under_hair] * 0.25))[:, None]
                sampled[under_hair] = np.clip(hair_all_med[None, :] * 0.78 * strand, 0.0, 255.0)
                has_wta[under_hair] = True

        if is_body:
            arm_to_wrist = params["wrist_x"] - params["sh_x"]
            for h_tag in (1, 2):
                is_hand = (q_tag == h_tag)
                if not is_hand.any(): continue
                sgn_h = 1.0 if h_tag == 1 else -1.0
                rot_a = 0.0 if params["is_tpose"] else (-(params["alpha_L"] if h_tag == 1 else params["alpha_R"]) * sgn_h)
                s_axis = (q_pos[:, 0] - sgn_h * params["sh_x"]) * (sgn_h * math.cos(rot_a)) + (q_pos[:, 1] - params["shoulder_y"]) * (sgn_h * math.sin(rot_a))
                dist_f = np.clip((s_axis - arm_to_wrist) / max(0.05, params["hand_len"]), 0.0, 1.0)
                palm_samp = is_hand & has_wta & (dist_f >= 0.30) & (dist_f <= 0.85) & (q_nrm[:, 1] > 0.05)
                if palm_samp.sum() < 12: palm_samp = is_hand & has_wta & (dist_f >= 0.22)
                if palm_samp.sum() < 8: palm_samp = is_hand & has_wta
                palm_rgb = np.median(sampled[palm_samp], axis=0).astype(np.float32) if palm_samp.any() else np.array([240.0, 212.0, 194.0], dtype=np.float32)
                crease = (0.93 + 0.07 * np.clip(0.5 + 0.5 * q_nrm[:, 1], 0.0, 1.0))[:, None]
                inlier = np.abs(sampled - palm_rgb[None, :]).sum(axis=1) <= 55.0
                w_keep = np.where(has_wta & inlier, 0.38 * (1.0 - dist_f), 0.0)[:, None]
                sampled[is_hand] = w_keep[is_hand] * sampled[is_hand] + (1.0 - w_keep[is_hand]) * (palm_rgb[None, :] * crease[is_hand])

            # Shade the closed hip cap as a plain dark matte surface
            is_pelvis = (q_tag == 4)
            if is_pelvis.any():
                pel_col = np.minimum(skirt_hem_rgb * 0.45, np.array([46.0, 50.0, 62.0], dtype=np.float32))
                pel_shade = (0.86 + 0.14 * np.clip(0.5 + 0.5 * q_nrm[is_pelvis, 1], 0.0, 1.0))[:, None]
                sampled[is_pelvis] = np.clip(pel_col[None, :] * pel_shade, 0.0, 255.0)
                has_wta[is_pelvis] = True

            upper_leg_miss = (q_tag == 0) & (~has_wta) & (q_pos[:, 1] < 0.48 * params["H"]) & (q_pos[:, 1] > 0.22 * params["H"])
            if upper_leg_miss.any():
                shin_ref = (q_tag == 0) & has_wta & (q_pos[:, 1] > 0.20 * params["H"]) & (q_pos[:, 1] < 0.34 * params["H"])
                if shin_ref.any():
                    sampled[upper_leg_miss] = np.median(sampled[shin_ref], axis=0)[None, :]

        tex_img = np.zeros((TEX_SZ, TEX_SZ, 3), dtype=np.uint8)
        tex_img[cov_ys, cov_xs] = np.clip(sampled, 0.0, 255.0).astype(np.uint8)

        if (~has_wta).any():
            valid_wta, need_inp = np.zeros((TEX_SZ, TEX_SZ), dtype=bool), np.zeros((TEX_SZ, TEX_SZ), dtype=bool)
            valid_wta[cov_ys[has_wta], cov_xs[has_wta]] = True
            need_inp[cov_ys[~has_wta], cov_xs[~has_wta]] = True
            if is_torso:
                lin_mask = np.zeros((TEX_SZ, TEX_SZ), dtype=bool)
                lin_mask[cov_ys[q_tag == 3], cov_xs[q_tag == 3]] = True
                if lin_mask.any():
                    tex_img[lin_mask] = np.clip(skirt_hem_rgb * 0.52, 0, 255).astype(np.uint8)
                    valid_wta[lin_mask] = True; need_inp[lin_mask] = False
                _inpaint_vertical_down(tex_img, valid_wta, need_inp, max_r=64)
            if is_body:
                is_hp = np.zeros((TEX_SZ, TEX_SZ), dtype=bool)
                is_hp[cov_ys[q_tag > 0], cov_xs[q_tag > 0]] = True
                need_inp &= (~is_hp)
            _inpaint_horizontal_gaps(tex_img, valid_wta, need_inp, max_r=(8 if is_torso else 64))

        tex_img = _despeckle_uv_outliers(tex_img, tex_cov, diff_thresh=80.0)
        tex_img = _despeckle_uv_outliers(tex_img, tex_cov, diff_thresh=80.0)
        part["tex"] = _dilate_uv_gutter(tex_img, tex_cov, passes=8)


# ---------------------------------------------------------------------------
# 6. Rigging, Morphs, & VRM 1.0 / 0.x Export
# ---------------------------------------------------------------------------

def _skin_part(pname: str, part: dict, bone_info: dict, params: dict):
    H, is_tpose, sh_x, shoulder_y = params["H"], params["is_tpose"], bone_info["sh_x"], bone_info["shoulder_y"]
    v = part["verts"].copy()
    Nv, W = len(v), np.zeros((len(v), len(BONES)), dtype=np.float32)
    wx, wy, wz = v[:, 0], v[:, 1], v[:, 2]
    yn = wy / H
    neck_y, chest_y, spine_y, hips_y, arm_len = bone_info["neck_y"], bone_info["chest_y"], bone_info["spine_y"], bone_info["hips_y"], bone_info["arm_len"]
    dir_L, dir_R, sh_L, sh_R = bone_info["dir_L"], bone_info["dir_R"], bone_info["sh_L"], bone_info["sh_R"]

    if pname == "head":
        t_hn = np.clip((wy - (neck_y - 0.025)) / 0.045, 0.0, 1.0)
        t_top = np.clip((wy - neck_y) / 0.035, 0.0, 1.0)
        W[:, BONE_IDX["chest"]] = 1.0 - t_hn; W[:, BONE_IDX["neck"]] = t_hn * (1.0 - t_top); W[:, BONE_IDX["head"]] = t_top
    elif pname == "hair":
        t_low = np.clip((neck_y - 0.02 - wy) / 0.14, 0.0, 1.0)
        W[:, BONE_IDX["head"]] = 1.0 - 0.35 * t_low; W[:, BONE_IDX["neck"]] = 0.20 * t_low; W[:, BONE_IDX["chest"]] = 0.15 * t_low
    elif pname == "torso":
        up = yn >= 0.49
        if up.any():
            uy, ux = wy[up], wx[up]
            w_nk = np.clip((uy - (neck_y - 0.020)) / 0.040, 0.0, 1.0)
            w_c = np.clip((uy - spine_y) / max(0.05, chest_y - spine_y), 0.0, 1.0) * (1.0 - w_nk)
            w_s = np.clip((uy - hips_y) / max(0.05, spine_y - hips_y), 0.0, 1.0) * (1.0 - w_c - w_nk)
            w_h = np.maximum(0.0, 1.0 - w_c - w_s - w_nk)
            sh_zone = np.clip((uy - (shoulder_y - 0.05)) / 0.05, 0.0, 1.0)
            sh_l = 0.25 * sh_zone * np.clip((ux - sh_x * 0.72) / 0.06, 0.0, 1.0)
            sh_r = 0.25 * sh_zone * np.clip((-ux - sh_x * 0.72) / 0.06, 0.0, 1.0)
            W[up, BONE_IDX["neck"]] = w_nk; W[up, BONE_IDX["chest"]] = w_c * (1.0 - sh_l - sh_r)
            W[up, BONE_IDX["leftShoulder"]] = w_c * sh_l; W[up, BONE_IDX["rightShoulder"]] = w_c * sh_r
            W[up, BONE_IDX["spine"]] = w_s; W[up, BONE_IDX["hips"]] = w_h
        lo = ~up
        if lo.any():
            sy, sx = yn[lo], wx[lo]
            t_hem = np.clip((0.49 - sy) / 0.18, 0.0, 1.0)
            l_f = np.clip((sx + 0.04) / 0.08, 0.0, 1.0)
            W[lo, BONE_IDX["hips"]] = 1.0 - 0.45 * t_hem
            W[lo, BONE_IDX["leftUpperLeg"]] = 0.45 * t_hem * l_f
            W[lo, BONE_IDX["rightUpperLeg"]] = 0.45 * t_hem * (1.0 - l_f)
    else:
        s_L = (wx - sh_L[0]) * dir_L[0] + (wy - sh_L[1]) * dir_L[1]
        d_perp_L = np.abs((wx - sh_L[0]) * (-dir_L[1]) + (wy - sh_L[1]) * dir_L[0])
        s_R = (wx - sh_R[0]) * dir_R[0] + (wy - sh_R[1]) * dir_R[1]
        d_perp_R = np.abs((wx - sh_R[0]) * (-dir_R[1]) + (wy - sh_R[1]) * dir_R[0])

        is_hand_L, is_hand_R, is_pelvis = (part["part_tag"] == 1), (part["part_tag"] == 2), (part["part_tag"] == 4)
        if is_pelvis.any():
            W[is_pelvis, BONE_IDX["hips"]] = 1.0
        if is_tpose:
            is_left_arm = is_hand_L | ((~is_pelvis) & (wx > sh_x - 0.04) & (np.abs(wy - shoulder_y) < 0.09) & (s_L > -0.04))
            is_right_arm = is_hand_R | ((~is_pelvis) & (wx < -(sh_x - 0.04)) & (np.abs(wy - shoulder_y) < 0.09) & (s_R > -0.04))
        else:
            is_left_arm = is_hand_L | ((~is_pelvis) & (wx > sh_x - 0.03) & (s_L > -0.04) & (d_perp_L < 0.085) & (wy <= shoulder_y + 0.08))
            is_right_arm = is_hand_R | ((~is_pelvis) & (wx < -(sh_x - 0.03)) & (s_R > -0.04) & (d_perp_R < 0.085) & (wy <= shoulder_y + 0.08))

        hand_s_start = (params["wrist_x"] - sh_x) / max(arm_len, 0.1)
        for side_mask, s_proj, bn_sh, bn_ua, bn_la, bn_hd, is_hd in (
            (is_left_arm, s_L, "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand", is_hand_L),
            (is_right_arm, s_R, "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand", is_hand_R),
        ):
            if side_mask.any():
                sl = s_proj[side_mask] / max(arm_len, 0.1)
                w_clav, w_sh, w_el = np.clip((sl + 0.06) / 0.06, 0.0, 1.0), np.clip((sl + 0.02) / 0.16, 0.0, 1.0), np.clip((sl - 0.42) / 0.12, 0.0, 1.0)
                w_wr = np.where(is_hd[side_mask], 1.0, np.clip((sl - (hand_s_start - 0.04)) / 0.04, 0.0, 1.0))
                W[side_mask, BONE_IDX["chest"]] = 1.0 - w_clav
                W[side_mask, BONE_IDX[bn_sh]] = w_clav * (1.0 - w_sh)
                W[side_mask, BONE_IDX[bn_ua]] = w_sh * (1.0 - w_el)
                W[side_mask, BONE_IDX[bn_la]] = w_el * (1.0 - w_wr)
                W[side_mask, BONE_IDX[bn_hd]] = w_wr

        legs = ~(is_left_arm | is_right_arm | is_pelvis)
        if legs.any():
            ly, lx, lz = wy[legs] / H, wx[legs], wz[legs]
            l_f = np.clip((lx + 0.035) / 0.07, 0.0, 1.0)
            w_leg, w_knee, w_ank = np.clip((0.49 - ly) / 0.10, 0.0, 1.0), np.clip((0.28 - ly) / 0.07, 0.0, 1.0), np.clip((0.07 - ly) / 0.03, 0.0, 1.0)
            w_toe = np.where(ly < 0.052, np.clip((lz - (params["cz_legs"] + 0.032)) / 0.045, 0.0, 1.0), 0.0)
            w_up, w_lo, w_ft = w_leg * (1.0 - w_knee), w_knee * (1.0 - w_ank), w_ank * (1.0 - w_toe)
            W[legs, BONE_IDX["hips"]] = 1.0 - w_leg
            W[legs, BONE_IDX["leftUpperLeg"]] = w_up * l_f; W[legs, BONE_IDX["rightUpperLeg"]] = w_up * (1.0 - l_f)
            W[legs, BONE_IDX["leftLowerLeg"]] = w_lo * l_f; W[legs, BONE_IDX["rightLowerLeg"]] = w_lo * (1.0 - l_f)
            W[legs, BONE_IDX["leftFoot"]] = w_ft * l_f; W[legs, BONE_IDX["rightFoot"]] = w_ft * (1.0 - l_f)
            W[legs, BONE_IDX["leftToes"]] = w_ank * w_toe * l_f; W[legs, BONE_IDX["rightToes"]] = w_ank * w_toe * (1.0 - l_f)

    W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-6)
    top4 = np.argsort(-W, axis=1)[:, :4]
    w4_raw = np.where(np.take_along_axis(W, top4, axis=1).astype(np.float64) > 1e-5, np.take_along_axis(W, top4, axis=1).astype(np.float64), 0.0)
    w4_raw /= np.maximum(w4_raw.sum(axis=1, keepdims=True), 1e-12)
    w4 = w4_raw.astype(np.float32); w4[:, 0] += (1.0 - w4.sum(axis=1))
    top4 = np.where(w4 > 0.0, top4, 0).astype(np.uint16)

    v_tpose = v.copy()
    if (pname == "body") and (not is_tpose) and (params["alpha_L"] > 1e-4 or params["alpha_R"] > 1e-4):
        w_arm_L = W[:, BONE_IDX["leftUpperArm"]] + W[:, BONE_IDX["leftLowerArm"]] + W[:, BONE_IDX["leftHand"]]
        w_arm_R = W[:, BONE_IDX["rightUpperArm"]] + W[:, BONE_IDX["rightLowerArm"]] + W[:, BONE_IDX["rightHand"]]
        theta_v = w_arm_L * params["alpha_L"] - w_arm_R * params["alpha_R"]
        is_l_piv = w_arm_L >= w_arm_R
        piv_x, piv_y = np.where(is_l_piv, sh_L[0], sh_R[0]), np.where(is_l_piv, sh_L[1], sh_R[1])
        c_v, s_v, has_arm = np.cos(theta_v), np.sin(theta_v), (w_arm_L + w_arm_R) > 1e-5
        dx_p, dy_p = v[has_arm, 0] - piv_x[has_arm], v[has_arm, 1] - piv_y[has_arm]
        v_tpose[has_arm, 0] = piv_x[has_arm] + c_v[has_arm] * dx_p - s_v[has_arm] * dy_p
        v_tpose[has_arm, 1] = piv_y[has_arm] + s_v[has_arm] * dx_p + c_v[has_arm] * dy_p

    return {"verts": v_tpose.astype(np.float32), "normals": compute_vertex_normals(v_tpose, part["tris"]), "uv": part["uv"], "tris": part["tris"], "joints": top4, "weights": w4, "tex": part["tex"]}


def build_facial_morph_targets(verts: np.ndarray, params: dict):
    Nv, y_chin, head_h, cz_head = len(verts), params["y_chin"], params["head_h"], params["hcz"]
    vx, vy, vz = verts[:, 0], verts[:, 1], verts[:, 2]
    face_mask = np.clip((vz - (cz_head + 0.020)) / 0.035, 0.0, 1.0) * np.exp(-((vx / 0.055) ** 4)) * np.clip((vy - (y_chin - 0.005)) / 0.025, 0.0, 1.0)
    y_mouth, y_jaw, y_eye, y_brow = y_chin + 0.22 * head_h, y_chin + 0.12 * head_h, y_chin + 0.48 * head_h, y_chin + 0.60 * head_h
    def gauss(cx, cy, sx, sy): return np.exp(-(((vx - cx) / sx) ** 2 + ((vy - cy) / sy) ** 2)) * face_mask
    g_mouth, g_lowlip, g_uplip, g_jaw = gauss(0, y_mouth, 0.024, 0.016), gauss(0, y_mouth - 0.008, 0.022, 0.018), gauss(0, y_mouth + 0.006, 0.020, 0.010), gauss(0, y_jaw, 0.032, 0.024)
    g_corners = gauss(0.018, y_mouth, 0.014, 0.012) + gauss(-0.018, y_mouth, 0.014, 0.012)
    g_cheek = gauss(0.028, y_mouth + 0.022, 0.022, 0.018) + gauss(-0.028, y_mouth + 0.022, 0.022, 0.018)
    g_eye_l_up, g_eye_l_dn = gauss(0.028, y_eye + 0.007, 0.016, 0.011), gauss(0.028, y_eye - 0.007, 0.016, 0.010)
    g_eye_r_up, g_eye_r_dn = gauss(-0.028, y_eye + 0.007, 0.016, 0.011), gauss(-0.028, y_eye - 0.007, 0.016, 0.010)
    g_brow, sgn_x = gauss(0.026, y_brow, 0.018, 0.012) + gauss(-0.026, y_brow, 0.018, 0.012), np.sign(vx)
    def mk(dx=0.0, dy=0.0, dz=0.0):
        d = np.zeros((Nv, 3), dtype=np.float32); d[:, 0] = dx; d[:, 1] = dy; d[:, 2] = dz; return d
    morphs = {
        "aa": mk(-0.002 * sgn_x * g_mouth, -0.010 * g_lowlip - 0.007 * g_jaw + 0.003 * g_uplip, -0.003 * g_jaw),
        "ih": mk(0.007 * sgn_x * g_corners, -0.003 * g_lowlip + 0.002 * g_uplip, 0.0),
        "ou": mk(-0.006 * sgn_x * g_corners, -0.005 * g_lowlip, 0.008 * g_mouth),
        "ee": mk(0.006 * sgn_x * g_corners, -0.005 * g_lowlip + 0.002 * g_uplip, 0.0),
        "oh": mk(-0.004 * sgn_x * g_corners, -0.008 * g_lowlip - 0.004 * g_jaw, 0.006 * g_mouth),
        "blinkLeft": mk(0.0, -0.0075 * g_eye_l_up + 0.0025 * g_eye_l_dn, 0.0015 * g_eye_l_up),
        "blinkRight": mk(0.0, -0.0075 * g_eye_r_up + 0.0025 * g_eye_r_dn, 0.0015 * g_eye_r_up),
    }
    morphs["blink"] = (morphs["blinkLeft"] + morphs["blinkRight"]).astype(np.float32)
    morphs["happy"] = mk(0.005 * sgn_x * g_corners, 0.006 * g_corners + 0.004 * g_cheek - 0.004 * (g_eye_l_up + g_eye_r_up) + 0.003 * (g_eye_l_dn + g_eye_r_dn), 0.003 * g_cheek)
    morphs["angry"] = mk(-0.003 * sgn_x * g_brow, -0.006 * g_brow - 0.003 * g_corners, 0.002 * g_brow)
    morphs["sad"] = mk(0.002 * sgn_x * g_brow, 0.004 * gauss(0.0, y_brow, 0.022, 0.012) - 0.005 * g_corners, 0.0)
    morphs["relaxed"] = mk(0.003 * sgn_x * g_corners, 0.003 * g_corners - 0.003 * (g_eye_l_up + g_eye_r_up), 0.0)
    morphs["surprised"] = mk(0.0, 0.007 * g_brow + 0.003 * (g_eye_l_up + g_eye_r_up) - 0.007 * g_lowlip, 0.002 * g_brow)
    return morphs


def rig_avatar(parts: dict, params: dict):
    H, sh_x, shoulder_y, cz_arm = params["H"], params["sh_x"], params["shoulder_y"], params["cz_arm"]
    neck_y = params["y_clavicle"] + 0.35 * (params["y_chin"] - params["y_clavicle"])
    cz_hips, cz_head, cz_legs = params["cz_torso"], params["hcz"], params["cz_legs"]
    arm_span, alpha_L, alpha_R = params["arm_tip_x"] - sh_x, params["alpha_L"], params["alpha_R"]
    bone_info = {
        "sh_x": sh_x, "shoulder_y": shoulder_y, "neck_y": neck_y,
        "chest_y": 0.69 * H, "spine_y": 0.59 * H, "hips_y": 0.50 * H, "arm_len": arm_span,
        "dir_L": np.array([math.cos(alpha_L), -math.sin(alpha_L), 0.0], dtype=np.float32),
        "dir_R": np.array([-math.cos(alpha_R), -math.sin(alpha_R), 0.0], dtype=np.float32),
        "sh_L": np.array([sh_x, shoulder_y, cz_arm], dtype=np.float32),
        "sh_R": np.array([-sh_x, shoulder_y, cz_arm], dtype=np.float32),
    }
    leg_x = float(params["leg_cx_L"][4])
    eye_y, head_y = params["y_chin"] + 0.48 * params["head_h"], params["y_chin"] + 0.30 * params["head_h"]
    elbow_x, hand_x = sh_x + 0.48 * (params["wrist_x"] - sh_x), params["wrist_x"]
    bone_pos = {
        "hips": (0.0, 0.50 * H, cz_hips), "spine": (0.0, 0.59 * H, cz_hips), "chest": (0.0, 0.69 * H, cz_hips),
        "neck": (0.0, neck_y, params["cz_neck"]), "head": (0.0, head_y, cz_head),
        "leftEye": (0.028, eye_y, cz_head + params["rz_head"] * 0.85), "rightEye": (-0.028, eye_y, cz_head + params["rz_head"] * 0.85),
        "leftShoulder": (sh_x * 0.45, shoulder_y - 0.008, cz_arm), "leftUpperArm": (sh_x, shoulder_y, cz_arm),
        "leftLowerArm": (elbow_x, shoulder_y, cz_arm), "leftHand": (hand_x, shoulder_y, cz_arm),
        "rightShoulder": (-sh_x * 0.45, shoulder_y - 0.008, cz_arm), "rightUpperArm": (-sh_x, shoulder_y, cz_arm),
        "rightLowerArm": (-elbow_x, shoulder_y, cz_arm), "rightHand": (-hand_x, shoulder_y, cz_arm),
        "leftUpperLeg": (leg_x, 0.46 * H, cz_legs), "leftLowerLeg": (leg_x, 0.25 * H, cz_legs), "leftFoot": (leg_x, 0.055 * H, cz_legs + 0.008), "leftToes": (leg_x, 0.014 * H, cz_legs + 0.075),
        "rightUpperLeg": (-leg_x, 0.46 * H, cz_legs), "rightLowerLeg": (-leg_x, 0.25 * H, cz_legs), "rightFoot": (-leg_x, 0.055 * H, cz_legs + 0.008), "rightToes": (-leg_x, 0.014 * H, cz_legs + 0.075),
    }
    rigged = {pn: _skin_part(pn, parts[pn], bone_info, params) for pn in ("body", "head", "torso", "hair") if parts.get(pn) is not None and len(parts[pn]["tris"]) > 0}
    rigged["head"]["morphs"] = build_facial_morph_targets(rigged["head"]["verts"], params)
    return rigged, bone_pos


def _png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True); return buf.getvalue()


def make_front_thumbnail(frames0: np.ndarray, fg0: np.ndarray, size: int = 256) -> Image.Image:
    ys, xs = np.nonzero(fg0)
    im = Image.fromarray(np.dstack([frames0, np.where(fg0, 255, 0).astype(np.uint8)]), "RGBA")
    if len(ys):
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        cy, half, cx = y0 + int(0.28 * (y1 - y0)), max(90, int(0.30 * (y1 - y0))), (x0 + x1) // 2
        crop = im.crop((max(0, cx - half), max(0, cy - half), min(frames0.shape[1], cx + half), min(frames0.shape[0], cy + half))).resize((size - 16, size - 16), Image.Resampling.LANCZOS)
    else:
        crop = im.resize((size - 16, size - 16), Image.Resampling.LANCZOS)
    out = Image.new("RGBA", (size, size), (28, 34, 58, 255))
    out.alpha_composite(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return out.convert("RGB")


def write_vrm_bytes(rigged: dict, bone_pos: dict, thumb_im: Image.Image,
                    name: str = "Avatar", author: str = "Unknown",
                    version: str = "1.0", shader_mode: str = "mtoon") -> bytes:
    bp = {k: tuple(float(c) for c in v) for k, v in bone_pos.items()}
    flip_vec = np.array([-1.0, 1.0, -1.0], dtype=np.float32)
    if version == "0.x":
        bp = {k: (-v[0], v[1], -v[2]) for k, v in bp.items()}

    active_names = [pn for pn in ("body", "head", "torso", "hair") if pn in rigged]
    binb, views, accs = bytearray(), [], []
    def add_view(data: bytes, target=None):
        while len(binb) % 4: binb.append(0)
        v = {"buffer": 0, "byteOffset": len(binb), "byteLength": len(data)}
        if target: v["target"] = target
        binb.extend(data); views.append(v); return len(views) - 1
    def add_acc(arr: np.ndarray, ctype: int, typ: str, target=None, minmax=False):
        a = {"bufferView": add_view(arr.tobytes(), target), "componentType": ctype, "count": int(arr.shape[0]), "type": typ}
        if minmax: a["min"] = [float(x) for x in arr.min(0)]; a["max"] = [float(x) for x in arr.max(0)]
        accs.append(a); return len(accs) - 1

    meshes_gltf = []
    for mi, pn in enumerate(active_names):
        part = rigged[pn]
        P, Nn = part["verts"].astype(np.float32).copy(), part["normals"].astype(np.float32).copy()
        if version == "0.x": P *= flip_vec; Nn *= flip_vec
        a_pos, a_nrm, a_uv = add_acc(P, 5126, "VEC3", 34962, True), add_acc(Nn, 5126, "VEC3", 34962), add_acc(part["uv"].astype(np.float32), 5126, "VEC2", 34962)
        a_j, a_w = add_acc(part["joints"].astype(np.uint16), 5123, "VEC4", 34962), add_acc(part["weights"].astype(np.float32), 5126, "VEC4", 34962)
        flat_idx = part["tris"].reshape(-1); big = int(flat_idx.max()) >= 65535
        a_i = add_acc(flat_idx.astype(np.uint32 if big else np.uint16).reshape(-1, 1), 5125 if big else 5123, "SCALAR", 34963)
        accs[a_i]["count"] = int(len(flat_idx))
        prim = {"attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv, "JOINTS_0": a_j, "WEIGHTS_0": a_w}, "indices": a_i, "material": mi, "mode": 4}
        mobj = {"name": pn.capitalize(), "primitives": [prim]}
        if pn == "head":
            tgts = []
            for ename in EXPR_NAMES:
                d = part["morphs"][ename].astype(np.float32).copy()
                if version == "0.x": d *= flip_vec
                tgts.append({"POSITION": add_acc(d, 5126, "VEC3", 34962, True)})
            prim["targets"], prim["extras"] = tgts, {"targetNames": list(EXPR_NAMES)}
            mobj["weights"], mobj["extras"] = [0.0] * len(EXPR_NAMES), {"targetNames": list(EXPR_NAMES)}
        meshes_gltf.append(mobj)

    bone_off = 1
    nodes = [{"name": "Root", "children": [bone_off + BONES.index("hips")]}]
    for bn in BONES:
        p, par = np.asarray(bp[bn], dtype=np.float64), PARENT[bn]
        nodes.append({"name": bn, "translation": [float(c) for c in (p - (np.asarray(bp[par], dtype=np.float64) if par else 0.0))]})
    for bn in BONES:
        kids = [bone_off + BONES.index(c) for c in BONES if PARENT[c] == bn]
        if kids: nodes[bone_off + BONES.index(bn)]["children"] = kids

    ibm = np.zeros((len(BONES), 16), dtype=np.float32)
    for k, bn in enumerate(BONES):
        M = np.eye(4, dtype=np.float32); M[:3, 3] = -np.asarray(bp[bn], dtype=np.float32); ibm[k] = M.T.reshape(-1)
    a_ibm = add_acc(ibm, 5126, "MAT4")

    part_nodes = {}
    for mi, pn in enumerate(active_names):
        part_nodes[pn] = len(nodes); nodes.append({"name": pn.capitalize(), "mesh": mi, "skin": 0})
    head_mesh_idx, head_node_idx = active_names.index("head"), part_nodes["head"]

    images_gltf, textures_gltf = [], []
    for mi, pn in enumerate(active_names):
        images_gltf.append({"name": f"{pn}_albedo", "bufferView": add_view(_png_bytes(Image.fromarray(rigged[pn]["tex"], "RGB"))), "mimeType": "image/png"})
        textures_gltf.append({"source": mi, "sampler": 0})
    thumb_idx = len(active_names)
    images_gltf.append({"name": "thumbnail", "bufferView": add_view(_png_bytes(thumb_im)), "mimeType": "image/png"})
    textures_gltf.append({"source": thumb_idx, "sampler": 1})
    while len(binb) % 4: binb.append(0)

    use_mtoon = (shader_mode == "mtoon")
    ext_used = ["VRMC_materials_mtoon"] if (version == "1.0" and use_mtoon) else ["KHR_materials_unlit"]
    def mk_mat(mat_name: str, tex_idx: int):
        mj = {"name": mat_name, "pbrMetallicRoughness": {"baseColorTexture": {"index": tex_idx}, "baseColorFactor": [1,1,1,1], "metallicFactor": 0.0, "roughnessFactor": 0.9 if use_mtoon else 1.0}, "emissiveFactor": [0.0, 0.0, 0.0], "alphaMode": "OPAQUE", "doubleSided": False, "extensions": {}}
        if version == "1.0" and use_mtoon:
            mj["extensions"]["VRMC_materials_mtoon"] = {"specVersion": "1.0", "transparentWithZWrite": False, "renderQueueOffsetNumber": 0, "shadeColorFactor": [0.85, 0.82, 0.87], "shadeMultiplyTexture": {"index": tex_idx}, "shadingShiftFactor": -0.15, "shadingToonyFactor": 0.45, "giEqualizationFactor": 0.85, "parametricRimColorFactor": [0.0, 0.0, 0.0], "rimLightingMixFactor": 1.0, "parametricRimFresnelPowerFactor": 5.0, "parametricRimLiftFactor": 0.0, "outlineWidthMode": "none", "outlineColorFactor": [0.0, 0.0, 0.0], "outlineLightingMixFactor": 1.0, "uvAnimationScrollXSpeedFactor": 0.0, "uvAnimationScrollYSpeedFactor": 0.0, "uvAnimationRotationSpeedFactor": 0.0}
        else:
            mj["extensions"]["KHR_materials_unlit"] = {}
        return mj

    j = {
        "asset": {"version": "2.0", "generator": "Video2VRM Studio (Zero-ML Polar-Loft Hybrid)"},
        "scene": 0, "scenes": [{"nodes": [0] + [part_nodes[pn] for pn in active_names]}],
        "nodes": nodes, "meshes": meshes_gltf,
        "skins": [{"joints": list(range(bone_off, bone_off + len(BONES))), "inverseBindMatrices": a_ibm, "skeleton": bone_off}],
        "materials": [mk_mat(f"{pn}_mat", mi) for mi, pn in enumerate(active_names)],
        "textures": textures_gltf, "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}, {"magFilter": 9729, "minFilter": 9729}],
        "images": images_gltf, "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": len(binb)}], "extensionsUsed": ext_used,
    }
    head_bone = bone_off + BONES.index("head")
    if version == "0.x":
        j["extensionsUsed"].append("VRM")
        bs_groups = [{"name": "Neutral", "presetName": "neutral", "binds": [], "materialValues": [], "isBinary": False}]
        for ti, ename in enumerate(EXPR_NAMES):
            bs_groups.append({"name": ename.capitalize(), "presetName": VRM0_EXPR_MAP[ename], "binds": [{"mesh": head_mesh_idx, "index": ti, "weight": 100}], "materialValues": [], "isBinary": False})
        def mk_v0_mat(mname: str, tid: int):
            return {"name": mname, "shader": "VRM/MToon" if use_mtoon else "VRM/UnlitTexture", "renderQueue": 2000, "floatProperties": {"_Cutoff": 0.5, "_BlendMode": 0.0, "_CullMode": 2.0, "_ZWrite": 1.0, "_SrcBlend": 1.0, "_DstBlend": 0.0, "_ReceiveShadowRate": 1.0, "_ShadingGradeRate": 1.0, "_ShadeShift": -0.15, "_ShadeToony": 0.45, "_IndirectLightIntensity": 0.85, "_RimLightingMix": 0.0, "_RimFresnelPower": 5.0, "_RimLift": 0.0, "_OutlineWidthMode": 0.0, "_OutlineWidth": 0.0, "_OutlineLightingMix": 1.0, "_MToonVersion": 38.0} if use_mtoon else {"_BlendMode": 0, "_Cutoff": 0.5}, "vectorProperties": {"_Color": [1,1,1,1], "_ShadeColor": [0.85,0.82,0.87,1], "_MainTex": [0,0,1,1], "_EmissionColor": [0,0,0,1], "_RimColor": [0,0,0,1], "_OutlineColor": [0,0,0,1]} if use_mtoon else {"_Color": [1,1,1,1], "_MainTex": [0,0,1,1]}, "textureProperties": {"_MainTex": tid, "_ShadeTexture": tid} if use_mtoon else {"_MainTex": tid}, "keywordMap": {}, "tagMap": {"RenderType": "Opaque"}}
        j["extensions"] = {"VRM": {"exporterVersion": "Video2VRM Studio", "specVersion": "0.0", "meta": {"title": name, "version": "1.0", "author": author, "contactInformation": "", "reference": "", "texture": thumb_idx, "allowedUserName": "Everyone", "violentUssageName": "Disallow", "sexualUssageName": "Disallow", "commercialUssageName": "Disallow", "otherPermissionUrl": "", "licenseName": "CC_BY_NC", "otherLicenseUrl": ""}, "humanoid": {"humanBones": [{"bone": bn, "node": bone_off + k, "useDefaultValues": True} for k, bn in enumerate(BONES)], "armStretch": 0.05, "legStretch": 0.05, "upperArmTwist": 0.5, "lowerArmTwist": 0.5, "upperLegTwist": 0.5, "lowerLegTwist": 0.5, "feetSpacing": 0, "hasTranslationDoF": False}, "firstPerson": {"firstPersonBone": head_bone, "firstPersonBoneOffset": {"x": 0, "y": 0.06, "z": 0}, "meshAnnotations": [{"mesh": mi, "firstPersonFlag": "ThirdPersonOnly" if pn in ("head", "hair") else "Both"} for mi, pn in enumerate(active_names)], "lookAtTypeName": "Bone", "lookAtHorizontalInner": {"curve": [0,0,0,1,1,1,1,0], "xRange": 90, "yRange": 10}, "lookAtHorizontalOuter": {"curve": [0,0,0,1,1,1,1,0], "xRange": 90, "yRange": 10}, "lookAtVerticalDown": {"curve": [0,0,0,1,1,1,1,0], "xRange": 90, "yRange": 10}, "lookAtVerticalUp": {"curve": [0,0,0,1,1,1,1,0], "xRange": 90, "yRange": 10}}, "blendShapeMaster": {"blendShapeGroups": bs_groups}, "secondaryAnimation": {"boneGroups": [], "colliderGroups": []}, "materialProperties": [mk_v0_mat(f"{pn}_mat", mi) for mi, pn in enumerate(active_names)]}}
    else:
        j["extensionsUsed"].append("VRMC_vrm")
        preset = {"neutral": {"isBinary": False, "overrideBlink": "none", "overrideLookAt": "none", "overrideMouth": "none"}}
        for ti, ename in enumerate(EXPR_NAMES):
            preset[ename] = {"morphTargetBinds": [{"node": head_node_idx, "index": ti, "weight": 1.0}], "isBinary": False, "overrideBlink": "none", "overrideLookAt": "none", "overrideMouth": "none"}
        curve_def = {"inputMaxValue": 90.0, "outputScale": 10.0}
        j["extensions"] = {"VRMC_vrm": {"specVersion": "1.0", "meta": {"name": name, "version": "1.0", "authors": [author], "licenseUrl": "https://vrm.dev/licenses/1.0/", "thumbnailImage": thumb_idx, "avatarPermission": "everyone", "commercialUsage": "personalNonProfit", "creditNotation": "required", "allowRedistribution": True, "modification": "allowModification", "allowExcessivelyViolentUsage": False, "allowExcessivelySexualUsage": False, "allowPoliticalOrReligiousUsage": False, "allowAntisocialOrHateUsage": False}, "humanoid": {"humanBones": {bn: {"node": bone_off + k} for k, bn in enumerate(BONES)}}, "firstPerson": {"meshAnnotations": [{"node": part_nodes[pn], "type": "thirdPersonOnly" if pn in ("head", "hair") else "both"} for pn in active_names]}, "lookAt": {"type": "bone", "offsetFromHeadBone": [0.0, 0.06, 0.0], "rangeMapHorizontalInner": curve_def, "rangeMapHorizontalOuter": curve_def, "rangeMapVerticalDown": curve_def, "rangeMapVerticalUp": curve_def}, "expressions": {"preset": preset}}}

    js = json.dumps(j, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(binb)
    return struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(binb), 0x004E4942) + bytes(binb)


def _read_glb(vrm_bytes: bytes):
    if len(vrm_bytes) < 20 or vrm_bytes[:4] != b"glTF":
        raise ValueError("Invalid glTF/GLB header")
    offset, j, bin_chunk = 12, None, b""
    while offset + 8 <= len(vrm_bytes):
        clen, ctype = struct.unpack("<II", vrm_bytes[offset:offset + 8])
        cdata = vrm_bytes[offset + 8:offset + 8 + clen]
        if ctype == 0x4E4F534A:
            j = json.loads(cdata.decode("utf-8"))
        elif ctype == 0x004E4942:
            bin_chunk = cdata
        offset += 8 + clen
    if j is None:
        raise ValueError("Missing JSON chunk in GLB")
    return j, bin_chunk


def _read_accessor(j: dict, bin_chunk: bytes, acc_idx: int) -> np.ndarray:
    acc = j["accessors"][acc_idx]
    bv = j["bufferViews"][acc["bufferView"]]
    comp_map = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    dim_map = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    dt, dims = np.dtype(comp_map[acc["componentType"]]), dim_map[acc["type"]]
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride")
    count = acc["count"]
    if stride and stride != dt.itemsize * dims:
        arr = np.empty((count, dims), dtype=dt)
        for i in range(count):
            s = start + i * stride
            arr[i] = np.frombuffer(bin_chunk, dtype=dt, count=dims, offset=s)
        return arr
    raw = np.frombuffer(bin_chunk, dtype=dt, count=count * dims, offset=start)
    return raw.reshape((count, dims)) if dims > 1 else raw.copy()


def validate_vrm_spec(vrm_bytes: bytes, expected_ver: str):
    ver_glb, total = struct.unpack("<II", vrm_bytes[4:12])
    assert vrm_bytes[:4] == b"glTF" and ver_glb == 2 and total == len(vrm_bytes) and (total % 4 == 0)
    j, binc = _read_glb(vrm_bytes)
    n_meshes = len(j["meshes"])
    assert 3 <= n_meshes <= 4 and len(j["materials"]) == n_meshes and len(j["meshes"][1].get("weights", [])) == len(EXPR_NAMES)
    for mobj in j["meshes"]:
        pr = mobj["primitives"][0]
        pos = _read_accessor(j, binc, pr["attributes"]["POSITION"])
        nrm = _read_accessor(j, binc, pr["attributes"]["NORMAL"])
        wgt = _read_accessor(j, binc, pr["attributes"]["WEIGHTS_0"])
        assert np.isfinite(pos).all() and np.isfinite(nrm).all() and np.isfinite(wgt).all()
        assert np.all(np.abs(np.linalg.norm(nrm, axis=1) - 1.0) < 1e-3) and np.all(np.abs(wgt.sum(axis=1) - 1.0) < 1e-3)
    if expected_ver == "1.0":
        vrm = j["extensions"]["VRMC_vrm"]
        assert set(vrm["humanoid"]["humanBones"].keys()) == set(BONES)
        assert "morphTargetBinds" not in vrm["expressions"]["preset"]["neutral"]
        assert "outputScale" in vrm["lookAt"]["rangeMapHorizontalInner"]
    else:
        vrm = j["extensions"]["VRM"]
        assert len(vrm["humanoid"]["humanBones"]) == len(BONES) and len(vrm["materialProperties"]) == n_meshes


def build_vrm_from_video(mp4_path: Path, name: str = "Avatar", author: str = "Unknown",
                         pose_mode: str = "auto", key_th: float = 24.0,
                         shader_mode: str = "mtoon", ffmpeg_bin: str = None):
    t0 = time.perf_counter()
    frames = extract_frames_720(mp4_path, ffmpeg_bin=ffmpeg_bin)
    fg, fg_color, clean_rgb = chroma_key_and_despill(frames, key_th=key_th)
    calib = calibrate_views(fg, pose_mode=pose_mode)
    arm_gaps = None if calib["is_tpose"] else extract_arm_gap_curves(fg[0], float(calib["cxs"][0]), calib["yb_med"], calib["px_per_m"])
    params = fit_body_params(fg, clean_rgb, calib, arm_gaps=arm_gaps)

    occ_video, gx, gy, gz, dx, dy, dz = build_video_occupancy(fg, calib, params)
    torso_part = build_polar_lofted_torso(occ_video, gx, gy, gz, dx, dy, dz, fg, clean_rgb, calib, params, arm_gaps=arm_gaps)
    hair_part = carve_color_consistent_hair(occ_video, torso_part["occ_torso_fit"], gx, gy, gz, dx, dy, dz, fg, clean_rgb, calib, params)
    parts = {
        "head": build_procedural_head(params),
        "body": build_procedural_limbs_and_hands(params),
        "torso": torso_part,
        "hair": hair_part,
    }
    bake_textures(parts, fg, fg_color, clean_rgb, calib, params)
    rigged, bone_pos = rig_avatar(parts, params)

    thumb_im = make_front_thumbnail(frames[0], fg[0])
    vrm1 = write_vrm_bytes(rigged, bone_pos, thumb_im, name, author, "1.0", shader_mode)
    vrm0 = write_vrm_bytes(rigged, bone_pos, thumb_im, name, author, "0.x", shader_mode)
    dt = time.perf_counter() - t0
    tri_counts = {pn: len(rigged[pn]["tris"]) for pn in rigged}
    info = {
        "verts": sum(len(rigged[pn]["verts"]) for pn in rigged), "tris": sum(tri_counts.values()), "tri_counts": tri_counts,
        "vrm1_kb": len(vrm1) / 1024.0, "vrm0_kb": len(vrm0) / 1024.0, "is_tpose": calib["is_tpose"],
        "hem_range": (float(torso_part["y_hem"].min()), float(torso_part["y_hem"].max())),
        "neck_cm": (params["y_chin"] - params["y_clavicle"]) * 100.0, "dt": dt,
    }
    return vrm1, vrm0, rigged, params, info


def main(argv=None):
    parser = argparse.ArgumentParser(description="Zero-ML Turntable Video -> Spec-Compliant .VRM (1.0 / 0.x) Converter")
    parser.add_argument("input_mp4", type=Path, help="Input chroma-keyed 360 turnaround MP4 video")
    parser.add_argument("-o", "--output", type=Path, default=Path("output.vrm"), help="Output .vrm path")
    parser.add_argument("--vrm-version", choices=("1.0", "0.x", "both"), default="1.0", help="VRM specification version")
    parser.add_argument("--shader", choices=("mtoon", "unlit"), default="mtoon", help="Material shader type")
    parser.add_argument("--pose", choices=("auto", "tpose", "apose"), default="auto", help="Rest pose detection mode")
    parser.add_argument("--key-threshold", type=float, default=24.0, help="Chroma-key green dominance threshold")
    parser.add_argument("--name", default=None, help="Avatar title in VRM metadata")
    parser.add_argument("--author", default="Unknown", help="Author name written in the VRM (your name)")
    parser.add_argument("--save-uv-dir", type=Path, default=None, help="Optional directory to save 1024x1024 PNG UV atlases")
    parser.add_argument("--ffmpeg", default=None, help="Path to ffmpeg executable (defaults to PATH lookup)")
    parser.add_argument("--no-validate", action="store_true", help="Skip built-in binary VRM spec self-check")
    args = parser.parse_args(argv)

    if not args.input_mp4.is_file():
        parser.error(f"Input video not found: {args.input_mp4}")

    avatar_name = args.name or args.input_mp4.stem
    vrm1, vrm0, rigged, params, info = build_vrm_from_video(
        args.input_mp4, name=avatar_name, author=args.author,
        pose_mode=args.pose, key_th=args.key_threshold,
        shader_mode=args.shader, ffmpeg_bin=args.ffmpeg,
    )

    if not args.no_validate:
        if args.vrm_version in ("1.0", "both"):
            validate_vrm_spec(vrm1, "1.0")
        if args.vrm_version in ("0.x", "both"):
            validate_vrm_spec(vrm0, "0.x")

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    if args.vrm_version == "1.0":
        out_path.write_bytes(vrm1); written.append(out_path)
    elif args.vrm_version == "0.x":
        out_path.write_bytes(vrm0); written.append(out_path)
    else:
        p1 = out_path.with_name(f"{out_path.stem}_v1{out_path.suffix or '.vrm'}")
        p0 = out_path.with_name(f"{out_path.stem}_v0{out_path.suffix or '.vrm'}")
        p1.write_bytes(vrm1); p0.write_bytes(vrm0); written.extend([p1, p0])

    if args.save_uv_dir is not None:
        args.save_uv_dir.mkdir(parents=True, exist_ok=True)
        for pn, pdata in rigged.items():
            Image.fromarray(pdata["tex"], "RGB").save(args.save_uv_dir / f"{pn}_uv.png")

    tc_str = ", ".join(f"{k}={v:,}" for k, v in info["tri_counts"].items())
    print(f"[Video2VRM] tris={info['tris']:,} ({tc_str}), pose={'T-pose' if info['is_tpose'] else 'A-pose'}, "
          f"neck={info['neck_cm']:.1f}cm, hem=[{info['hem_range'][0]:.3f}..{info['hem_range'][1]:.3f}m], time={info['dt']:.2f}s")
    for wp in written:
        print(f"Saved -> {wp} ({wp.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
