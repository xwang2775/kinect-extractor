"""
Multi-person pose extraction with appearance-aided ID tracking.

Tracking cost combines two signals:
  position  — Hungarian matching on key joint positions (normalized 0-1)
  color     — HSV histogram of each person's torso region

  cost(track_i, det_j) = ALPHA * pos_dist + (1-ALPHA) * color_dist

Person color histograms are updated each frame with exponential moving average
so that gradual illumination changes don't break the model, but sudden appearance
changes (clothes swap) are handled correctly.
"""

import argparse
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pandas as pd
import os
import sys
import urllib.request
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

# ─── Defaults (overridable via CLI) ───────────────────────────────────────────
INPUT_FILE   = "input.mkv"
OUTPUT_DIR   = "output"
MODEL_PATH   = "pose_landmarker_full.task"
N_PERSONS    = 3     # max simultaneous persons to track
SKIP_FRAMES  = 1     # 1 = every frame
MAX_FRAMES   = None  # None = all
SAVE_VIDEO   = True

# Cost weights  (must sum to 1.0)
ALPHA        = 0.55  # weight for joint-position distance
# (1-ALPHA) is the weight for torso-color distance

MAX_COST     = 0.5   # reject match if combined cost exceeds this
COLOR_EMA    = 0.75  # exponential moving average for color model (closer to 1 = more inertia)

# HSV histogram bins
H_BINS = 18   # hue: 0-180 in OpenCV, 18 bins = 10° each
S_BINS = 8    # saturation: 0-255, 8 bins
HIST_DIM = H_BINS * S_BINS   # 144-D descriptor

# Torso region padding (fraction of bounding-box size added on each side)
TORSO_PAD = 0.15
# ─────────────────────────────────────────────────────────────────────────────

LANDMARK_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer",
    "right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_pinky","right_pinky",
    "left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee",
    "left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index"
]

# Landmark indices used for position matching (stable body-center joints)
KEY_JOINTS = [0, 11, 12, 13, 14, 23, 24, 25, 26]

# Landmark indices that define the torso bounding box
TORSO_JOINTS = [11, 12, 23, 24]   # L-shoulder, R-shoulder, L-hip, R-hip

# Per-person drawing colors (BGR)
PERSON_COLORS = [
    (0,   220,   0),   # green   — person 0
    (0,   120, 255),   # orange  — person 1
    (220,   0, 220),   # magenta — person 2
    (255, 210,   0),   # cyan    — person 3
    (0,   220, 255),   # yellow  — person 4
]

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),
    (24,26),(26,28),(28,30),(28,32),
]


# ─── Feature extraction ───────────────────────────────────────────────────────

def lms_to_pos_array(lms) -> np.ndarray:
    """(x, y) of KEY_JOINTS as a 1-D float array."""
    return np.array([[lms[i].x, lms[i].y] for i in KEY_JOINTS], dtype=np.float32).ravel()


def extract_torso_hist(bgr_frame: np.ndarray, lms, img_h: int, img_w: int) -> np.ndarray | None:
    """
    Crop the torso bounding box (shoulders + hips) from the BGR frame and
    compute a normalized HSV histogram (H_BINS × S_BINS = 144-D).
    Returns None if the crop is invalid or any landmark has low visibility.
    """
    # Require all torso landmarks to be reasonably visible
    if any(lms[i].visibility < 0.4 for i in TORSO_JOINTS):
        return None

    xs = [lms[i].x for i in TORSO_JOINTS]
    ys = [lms[i].y for i in TORSO_JOINTS]

    bw = (max(xs) - min(xs)) * img_w
    bh = (max(ys) - min(ys)) * img_h

    x0 = max(0, int((min(xs) - TORSO_PAD) * img_w))
    y0 = max(0, int((min(ys) - TORSO_PAD) * img_h))
    x1 = min(img_w, int((max(xs) + TORSO_PAD) * img_w))
    y1 = min(img_h, int((max(ys) + TORSO_PAD) * img_h))

    if x1 - x0 < 10 or y1 - y0 < 10:
        return None

    patch = bgr_frame[y0:y1, x0:x1]
    hsv   = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv], [0, 1], None,
        [H_BINS, S_BINS],
        [0, 180, 0, 256]
    ).ravel().astype(np.float32)

    total = hist.sum()
    if total < 1:
        return None
    return hist / total   # normalized to sum=1


def color_dist(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Chi-squared histogram distance, normalized to [0, 1].
    0 = identical histograms.
    """
    denom = h1 + h2
    mask  = denom > 1e-9
    chi2  = 0.5 * np.sum(((h1[mask] - h2[mask]) ** 2) / denom[mask])
    # chi2 is in [0, HIST_DIM/2]; normalize to [0, 1] by capping at 1
    return float(min(chi2 / (HIST_DIM * 0.5), 1.0))


# ─── Tracker ──────────────────────────────────────────────────────────────────

class PersonTracker:
    """
    Maintains person IDs across frames.

    State per track:
      pos_arr   — key-joint position array (updated every frame)
      color_hist — torso HSV histogram (updated via EMA)
    """

    def __init__(self):
        self.pos_arrs    = {}   # pid -> np.ndarray
        self.color_hists = {}   # pid -> np.ndarray | None
        self.next_id     = 0

    def _assign_new_id(self, lms, frame_bgr, img_h, img_w):
        pid  = self.next_id
        self.pos_arrs[pid]    = lms_to_pos_array(lms)
        self.color_hists[pid] = extract_torso_hist(frame_bgr, lms, img_h, img_w)
        self.next_id += 1
        return pid

    def update(self, detections, frame_bgr: np.ndarray, img_h: int, img_w: int):
        """
        Match detections to existing tracks using position + color cost.
        Returns list of (person_id, landmark_list) for this frame.
        """
        if not detections:
            return []

        if not self.pos_arrs:
            # First detections ever — assign IDs directly
            assigned = []
            for lms in detections[:N_PERSONS]:
                pid = self._assign_new_id(lms, frame_bgr, img_h, img_w)
                assigned.append((pid, lms))
            return assigned

        track_ids = list(self.pos_arrs.keys())
        n_t       = len(track_ids)
        n_d       = len(detections)

        # Build combined cost matrix  [n_tracks × n_detections]
        C = np.full((n_t, n_d), fill_value=1e6, dtype=np.float64)
        for ti, pid in enumerate(track_ids):
            for di, lms in enumerate(detections):
                # Position cost (mean joint distance, normalized)
                det_pos   = lms_to_pos_array(lms)
                pos_d     = np.linalg.norm(self.pos_arrs[pid] - det_pos) / len(KEY_JOINTS)

                # Color cost (chi-squared histogram distance)
                det_hist  = extract_torso_hist(frame_bgr, lms, img_h, img_w)
                trk_hist  = self.color_hists[pid]

                if det_hist is not None and trk_hist is not None:
                    col_d = color_dist(trk_hist, det_hist)
                else:
                    col_d = 0.5   # neutral penalty when histogram unavailable

                C[ti, di] = ALPHA * pos_d + (1.0 - ALPHA) * col_d

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(C)

        assigned     = []
        matched_dets = set()

        for ri, ci in zip(row_ind, col_ind):
            if C[ri, ci] > MAX_COST:
                continue   # too far — treat as unmatched
            pid = track_ids[ri]
            lms = detections[ci]

            # Update position (immediate)
            self.pos_arrs[pid] = lms_to_pos_array(lms)

            # Update color histogram (exponential moving average)
            new_hist = extract_torso_hist(frame_bgr, lms, img_h, img_w)
            if new_hist is not None:
                if self.color_hists[pid] is None:
                    self.color_hists[pid] = new_hist
                else:
                    self.color_hists[pid] = (COLOR_EMA * self.color_hists[pid]
                                             + (1.0 - COLOR_EMA) * new_hist)

            assigned.append((pid, lms))
            matched_dets.add(ci)

        # Unmatched detections → new IDs (up to N_PERSONS)
        for ci, lms in enumerate(detections):
            if ci not in matched_dets and len(self.pos_arrs) < N_PERSONS:
                pid = self._assign_new_id(lms, frame_bgr, img_h, img_w)
                assigned.append((pid, lms))

        return assigned


# ─── Drawing ──────────────────────────────────────────────────────────────────

def draw_person(frame, lms, color, img_h, img_w, pid):
    pts = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in lms]
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, color, -1)
    # Label above nose
    nose_x, nose_y = pts[0]
    cv2.putText(frame, f"P{pid}", (nose_x - 20, nose_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# ─── Main extraction loop ─────────────────────────────────────────────────────

def run(cap, fps, total, img_w, img_h, out_dir, model_path):
    """Detection pass — collects records only; overlay video is written later
    so its labels can match the relabeled (left/right) person IDs."""
    os.makedirs(out_dir, exist_ok=True)

    base_opts  = mp_python.BaseOptions(model_asset_path=model_path)
    mp_options = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=N_PERSONS,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(mp_options)
    tracker    = PersonTracker()

    all_records = []
    frame_idx   = 0
    processed   = 0
    limit       = MAX_FRAMES or total

    with tqdm(total=min(limit, total), desc="Tracking persons") as pbar:
        while cap.isOpened() and processed < limit:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            if frame_idx % SKIP_FRAMES != 0:
                frame_idx += 1
                continue

            ts_ms    = int(frame_idx / fps * 1000)
            rgb      = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect_for_video(mp_img, ts_ms)

            detections  = result.pose_landmarks or []
            assignments = tracker.update(detections, bgr_frame, img_h, img_w)

            ts_sec = round(frame_idx / fps, 4)

            for pid, lms in assignments:
                row = {"frame": frame_idx, "timestamp_sec": ts_sec, "person_id": pid}
                for i, name in enumerate(LANDMARK_NAMES):
                    lm = lms[i]
                    row[f"{name}_x"]   = round(lm.x, 5)
                    row[f"{name}_y"]   = round(lm.y, 5)
                    row[f"{name}_z"]   = round(lm.z, 5)
                    row[f"{name}_vis"] = round(lm.visibility, 4)
                all_records.append(row)

            frame_idx += 1
            processed += 1
            pbar.update(1)

    landmarker.close()
    return all_records


def render_overlay_video(input_path, df, out_dir, img_w, img_h, fps):
    """Second pass: re-decode video and draw using the relabeled person IDs."""
    if df.empty:
        return
    out_fps = fps / max(SKIP_FRAMES, 1)
    vpath   = os.path.join(out_dir, "multiperson_pose.mp4")
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(vpath, fourcc, out_fps, (img_w, img_h))
    print(f"[INFO] Rendering overlay video: {vpath}")

    has_side = "side" in df.columns
    frames_to_render = sorted(df["frame"].unique())
    by_frame = {f: g for f, g in df.groupby("frame")}
    max_frame = frames_to_render[-1]

    cap = cv2.VideoCapture(input_path)
    frame_idx = 0
    with tqdm(total=max_frame + 1, desc="Rendering overlay") as pbar:
        while cap.isOpened() and frame_idx <= max_frame:
            ret, bgr = cap.read()
            if not ret:
                break
            if frame_idx in by_frame:
                grp = by_frame[frame_idx]
                for _, row in grp.iterrows():
                    pid = int(row["person_id"])
                    color = PERSON_COLORS[pid % len(PERSON_COLORS)]
                    label = f"P{pid}({row['side']})" if has_side else f"P{pid}"
                    pts = [(int(row[f"{n}_x"] * img_w), int(row[f"{n}_y"] * img_h))
                           for n in LANDMARK_NAMES]
                    for a, b in POSE_CONNECTIONS:
                        cv2.line(bgr, pts[a], pts[b], color, 2)
                    for x, y in pts:
                        cv2.circle(bgr, (x, y), 5, color, -1)
                    nose_x, nose_y = pts[0]
                    cv2.putText(bgr, label, (nose_x - 30, nose_y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                writer.write(bgr)
            else:
                # No detections for this frame, but still write it (when SKIP_FRAMES==1)
                if SKIP_FRAMES == 1:
                    writer.write(bgr)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()


# ─── Output ───────────────────────────────────────────────────────────────────

def relabel_left_right(df):
    """
    Remap person_id so 0=leftmost, 1=next-to-left, ... based on each
    person's mean nose_x across the whole recording. Also adds a
    `side` column ("left"/"right") when exactly 2 persons are present.
    """
    if df.empty or "person_id" not in df.columns:
        return df
    mean_x = df.groupby("person_id")["nose_x"].mean().sort_values()
    old_to_new = {old_pid: new_pid for new_pid, old_pid in enumerate(mean_x.index)}
    df = df.copy()
    df["person_id"] = df["person_id"].map(old_to_new)
    df = df.sort_values(["frame", "person_id"]).reset_index(drop=True)

    n_persons = df["person_id"].nunique()
    if n_persons == 2:
        df["side"] = df["person_id"].map({0: "left", 1: "right"})
    return df


def save_results(records, out_dir):
    df = pd.DataFrame(records)

    # Remap IDs so 0=leftmost, 1=next, ... (stable across runs)
    df = relabel_left_right(df)

    # Combined CSV
    csv_all = os.path.join(out_dir, "multiperson_joints.csv")
    df.to_csv(csv_all, index=False)
    print(f"\n[OK] Combined CSV ({len(df)} rows): {csv_all}")

    # Per-person CSVs (named by side when n=2, otherwise by pid)
    has_side = "side" in df.columns
    for pid, grp in df.groupby("person_id"):
        if has_side:
            side = grp["side"].iloc[0]
            csv_p = os.path.join(out_dir, f"person_{pid}_{side}_joints.csv")
            label = f"Person {pid} ({side})"
        else:
            csv_p = os.path.join(out_dir, f"person_{pid}_joints.csv")
            label = f"Person {pid}"
        grp.to_csv(csv_p, index=False)
        n = len(grp)
        pct = 100 * n / (df["frame"].nunique() or 1)
        print(f"     {label}: {n} frames ({pct:.0f}% presence) → {csv_p}")

    # ── Plot 1: nose X/Y trajectory per person ────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for pid, grp in df.groupby("person_id"):
        c = [x/255 for x in PERSON_COLORS[pid % len(PERSON_COLORS)][::-1]]
        lbl = f"P{pid} ({grp['side'].iloc[0]})" if has_side else f"Person {pid}"
        axes[0].plot(grp["timestamp_sec"], grp["nose_x"], label=lbl, lw=0.8, color=c)
        axes[1].plot(grp["timestamp_sec"], grp["nose_y"], label=lbl, lw=0.8, color=c)
    axes[0].set_title("Nose X per person  (0=left, 1=right)")
    axes[1].set_title("Nose Y per person  (0=top,  1=bottom)")
    axes[1].set_xlabel("Time (sec)")
    for ax in axes:
        ax.legend(fontsize=9)
    plt.tight_layout()
    traj_path = os.path.join(out_dir, "multiperson_trajectories.png")
    plt.savefig(traj_path, dpi=150)
    print(f"[OK] Trajectory plot: {traj_path}")
    plt.close()

    # ── Plot 2: wrist speed per person (motion intensity) ─────────────────
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    fps_approx = 15.0
    for pid, grp in df.groupby("person_id"):
        c = [x/255 for x in PERSON_COLORS[pid % len(PERSON_COLORS)][::-1]]
        prefix = f"P{pid}({grp['side'].iloc[0]})" if has_side else f"P{pid}"
        for hand in ("left", "right"):
            dx = grp[f"{hand}_wrist_x"].diff()
            dy = grp[f"{hand}_wrist_y"].diff()
            spd = np.sqrt(dx**2 + dy**2) * fps_approx
            ax2.plot(grp["timestamp_sec"], spd,
                     label=f"{prefix} {hand} wrist", lw=0.7, color=c,
                     linestyle="-" if hand == "left" else "--")
    ax2.set_title("Wrist Speed per Person  (normalized coord/sec)")
    ax2.set_xlabel("Time (sec)")
    ax2.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    spd_path = os.path.join(out_dir, "multiperson_wrist_speed.png")
    plt.savefig(spd_path, dpi=150)
    print(f"[OK] Wrist speed plot: {spd_path}")
    plt.close()

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract multi-person pose data with appearance-aided ID tracking.")
    parser.add_argument("input", nargs="?", default=INPUT_FILE,
                        help="Input .mkv file (default: %(default)s)")
    parser.add_argument("-o", "--output", default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--model", default=MODEL_PATH,
                        help="Path to MediaPipe pose_landmarker_full.task (auto-downloaded if missing)")
    parser.add_argument("--n-persons", type=int, default=N_PERSONS,
                        help="Max simultaneous persons to track (default: %(default)d)")
    parser.add_argument("--skip-frames", type=int, default=SKIP_FRAMES,
                        help="Process every Nth frame (default: %(default)d)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit total frames processed (default: all)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip writing the multi-person overlay video")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="Weight for joint-position cost in matching (default: %(default).2f)")
    args = parser.parse_args()

    INPUT_FILE  = args.input
    OUTPUT_DIR  = args.output
    MODEL_PATH  = args.model
    N_PERSONS   = args.n_persons
    SKIP_FRAMES = args.skip_frames
    MAX_FRAMES  = args.max_frames
    SAVE_VIDEO  = not args.no_video
    ALPHA       = args.alpha

    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] {INPUT_FILE} not found"); sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Downloading pose model (~25 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[INFO] Saved to: {MODEL_PATH}")

    print(f"Cost function: {ALPHA:.0%} position + {1-ALPHA:.0%} torso-color")
    print(f"Max cost threshold: {MAX_COST}  |  Color EMA: {COLOR_EMA}\n")

    cap   = cv2.VideoCapture(INPUT_FILE)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] {w}x{h} @ {fps} fps, {total} frames (~{total/fps:.0f} sec)")

    records = run(cap, fps, total, w, h, OUTPUT_DIR, MODEL_PATH)
    cap.release()

    df = save_results(records, OUTPUT_DIR)
    if SAVE_VIDEO:
        render_overlay_video(INPUT_FILE, df, OUTPUT_DIR, w, h, fps)
    print(f"\n[Done] All files in ./{OUTPUT_DIR}/")
    print(f"  multiperson_joints.csv         — all persons, all frames")
    print(f"  person_N_joints.csv            — per-person data")
    print(f"  multiperson_pose.mp4           — color-coded skeleton video")
    print(f"  multiperson_trajectories.png   — nose position over time")
    print(f"  multiperson_wrist_speed.png    — wrist motion intensity")
