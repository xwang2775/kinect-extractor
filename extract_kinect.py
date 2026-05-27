"""
Kinect MKV Human Motion Data Extraction
Uses OpenCV to read color video stream + MediaPipe Tasks API for 33 body landmark detection.
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

# ─── Defaults (overridable via CLI) ───────────────────────────────────────────
INPUT_FILE  = "input.mkv"
OUTPUT_DIR  = "output"
MODEL_PATH  = "pose_landmarker_full.task"   # downloaded automatically if missing
SKIP_FRAMES = 1       # process every N frames (1=all, 2=every other, etc.)
MAX_FRAMES  = None    # None=all frames; set integer to limit (useful for testing)
SAVE_VIDEO  = True    # whether to write pose-overlay video
MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
# ─────────────────────────────────────────────────────────────────────────────

# MediaPipe 33 landmark names (in order)
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# Skeleton connections for drawing (pairs of landmark indices)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),    # face
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),     # left arm
    (12,14),(14,16),(16,18),(16,20),(16,22),             # right arm
    (11,23),(12,24),(23,24),                             # torso
    (23,25),(25,27),(27,29),(27,31),                     # left leg
    (24,26),(26,28),(28,30),(28,32),                     # right leg
]


def download_model(path, url):
    """Download the MediaPipe pose landmarker model file if not present."""
    if os.path.exists(path):
        print(f"[INFO] Model found: {path}")
        return
    print(f"[INFO] Downloading pose model (~25 MB) ...")
    print(f"       {url}")
    urllib.request.urlretrieve(url, path, reporthook=lambda b, bs, t: None)
    print(f"[INFO] Saved to: {path}")


def open_video(path):
    """Open MKV file and return capture object + metadata."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open file: {path}")
        sys.exit(1)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] File: {path}")
    print(f"[INFO] Resolution: {w}x{h},  FPS: {fps:.1f},  Total frames: {total}")
    return cap, fps, total, w, h


def draw_skeleton(frame, landmarks, h, w):
    """Draw landmark points and connections onto a BGR frame."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for (a, b) in POSE_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 80, 255), -1)
    return frame


def extract_pose(cap, fps, total, w, h, out_dir, model_path):
    """Main extraction loop: run MediaPipe on each frame, collect joint coordinates."""
    os.makedirs(out_dir, exist_ok=True)

    # Build MediaPipe pose landmarker (VIDEO mode = uses temporal smoothing)
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    options   = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=1,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    records = []   # list of dicts, one per processed frame

    # Optional output video writer
    writer = None
    if SAVE_VIDEO:
        out_fps   = fps / max(SKIP_FRAMES, 1)
        vout_path = os.path.join(out_dir, "pose_overlay.mp4")
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
        writer    = cv2.VideoWriter(vout_path, fourcc, out_fps, (w, h))
        print(f"[INFO] Output video: {vout_path}")

    limit     = MAX_FRAMES if MAX_FRAMES else total
    frame_idx = 0
    processed = 0

    with tqdm(total=min(limit, total), desc="Extracting pose") as pbar:
        while cap.isOpened() and processed < limit:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames if configured
            if frame_idx % SKIP_FRAMES != 0:
                frame_idx += 1
                continue

            timestamp_ms = int(frame_idx / fps * 1000)

            # Convert BGR -> RGB and wrap in MediaPipe Image
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            row = {
                "frame":         frame_idx,
                "timestamp_sec": round(frame_idx / fps, 4),
                "detected":      False,
            }

            if result.pose_landmarks:
                row["detected"] = True
                lms = result.pose_landmarks[0]   # first detected person
                for i, name in enumerate(LANDMARK_NAMES):
                    lm = lms[i]
                    row[f"{name}_x"]   = round(lm.x, 5)
                    row[f"{name}_y"]   = round(lm.y, 5)
                    row[f"{name}_z"]   = round(lm.z, 5)  # depth (normalized)
                    row[f"{name}_vis"] = round(lm.visibility, 4)

                if writer:
                    frame = draw_skeleton(frame, lms, h, w)

            records.append(row)

            if writer:
                writer.write(frame)

            frame_idx += 1
            processed += 1
            pbar.update(1)

    landmarker.close()
    if writer:
        writer.release()

    return records


def save_results(records, out_dir, fps):
    """Save CSV and generate overview plots."""
    df = pd.DataFrame(records)

    # Save joint positions to CSV
    csv_path = os.path.join(out_dir, "joint_positions.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Joint data saved: {csv_path}")
    print(f"     Processed frames: {len(df)},  Detections: {df['detected'].sum()}")

    det_df = df[df["detected"]].copy()

    # ── Plot 1: detection timeline + wrist trajectories + shoulder width ────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    det_flag = df["detected"].astype(int)
    axes[0].fill_between(df["timestamp_sec"], det_flag, alpha=0.4, color="steelblue")
    axes[0].set_title("Pose Detection Status  (1=detected, 0=not detected)")
    axes[0].set_xlabel("Time (sec)")
    axes[0].set_ylim(-0.1, 1.3)
    axes[0].set_yticks([0, 1])

    if not det_df.empty and "left_wrist_x" in det_df.columns:
        axes[1].plot(det_df["timestamp_sec"], det_df["left_wrist_x"],  label="Left wrist X",  lw=0.8)
        axes[1].plot(det_df["timestamp_sec"], det_df["right_wrist_x"], label="Right wrist X", lw=0.8)
        axes[1].plot(det_df["timestamp_sec"], det_df["left_wrist_y"],  label="Left wrist Y",  lw=0.8, ls="--")
        axes[1].plot(det_df["timestamp_sec"], det_df["right_wrist_y"], label="Right wrist Y", lw=0.8, ls="--")
        axes[1].set_title("Wrist Joint Trajectories (normalized image coords)")
        axes[1].set_xlabel("Time (sec)")
        axes[1].legend(fontsize=8)

    if not det_df.empty and "left_shoulder_x" in det_df.columns:
        shoulder_w = (det_df["left_shoulder_x"] - det_df["right_shoulder_x"]).abs()
        axes[2].plot(det_df["timestamp_sec"], shoulder_w, color="coral", lw=0.8)
        axes[2].set_title("Shoulder Width Over Time (reflects body orientation / distance)")
        axes[2].set_xlabel("Time (sec)")

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "motion_overview.png")
    plt.savefig(plot_path, dpi=150)
    print(f"[OK] Overview plot saved: {plot_path}")
    plt.close()

    # ── Plot 2: joint speed (motion intensity) ─────────────────────────────
    if not det_df.empty and "left_wrist_x" in det_df.columns:
        key_joints = ["left_wrist", "right_wrist", "left_ankle", "right_ankle", "nose"]
        fig2, ax2  = plt.subplots(figsize=(14, 4))
        for j in key_joints:
            dx    = det_df[f"{j}_x"].diff()
            dy    = det_df[f"{j}_y"].diff()
            speed = np.sqrt(dx**2 + dy**2) * fps   # approx speed in normalized units/sec
            ax2.plot(det_df["timestamp_sec"], speed, label=j, lw=0.7)
        ax2.set_title("Key Joint Speed (normalized coords/sec) — motion intensity indicator")
        ax2.set_xlabel("Time (sec)")
        ax2.legend(fontsize=8)
        plt.tight_layout()
        speed_path = os.path.join(out_dir, "joint_speed.png")
        plt.savefig(speed_path, dpi=150)
        print(f"[OK] Speed plot saved: {speed_path}")
        plt.close()

    return df


def print_summary(df):
    det_df = df[df["detected"]]
    print("\n═══════════════ Summary ═══════════════")
    print(f"  Video duration:   {df['timestamp_sec'].max():.1f} sec")
    print(f"  Frames processed: {len(df)}")
    print(f"  Frames detected:  {len(det_df)} ({100*len(det_df)/len(df):.1f}%)")
    if not det_df.empty and "nose_x" in det_df.columns:
        print(f"  Nose X mean:      {det_df['nose_x'].mean():.3f}  (0=left edge, 1=right edge)")
        print(f"  Nose Y mean:      {det_df['nose_y'].mean():.3f}  (0=top, 1=bottom)")
    print("═══════════════════════════════════════")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract single-person 33-landmark pose data from a Kinect MKV.")
    parser.add_argument("input", nargs="?", default=INPUT_FILE,
                        help="Input .mkv file (default: %(default)s)")
    parser.add_argument("-o", "--output", default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--model", default=MODEL_PATH,
                        help="Path to MediaPipe pose_landmarker_full.task (auto-downloaded if missing)")
    parser.add_argument("--skip-frames", type=int, default=SKIP_FRAMES,
                        help="Process every Nth frame (default: %(default)d)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit total frames processed (default: all)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip writing the pose-overlay video")
    args = parser.parse_args()

    INPUT_FILE  = args.input
    OUTPUT_DIR  = args.output
    MODEL_PATH  = args.model
    SKIP_FRAMES = args.skip_frames
    MAX_FRAMES  = args.max_frames
    SAVE_VIDEO  = not args.no_video

    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File not found: {INPUT_FILE}")
        sys.exit(1)

    download_model(MODEL_PATH, MODEL_URL)
    cap, fps, total, w, h = open_video(INPUT_FILE)
    records = extract_pose(cap, fps, total, w, h, OUTPUT_DIR, MODEL_PATH)
    cap.release()

    df = save_results(records, OUTPUT_DIR, fps)
    print_summary(df)

    print(f"\n[Done] All outputs written to ./{OUTPUT_DIR}/")
