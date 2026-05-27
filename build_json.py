"""
Merge all extracted data into a structured JSON file.
Also extracts IMU data from Stream 3 and pose_world_landmarks (3D in meters).
"""

import argparse
import av
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pandas as pd
import struct
import json
import os
import sys
import urllib.request
from tqdm import tqdm

MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

# ─── Defaults (overridable via CLI) ───────────────────────────────────────────
INPUT_FILE  = "input.mkv"
OUTPUT_DIR  = "output"
MODEL_PATH  = "pose_landmarker_full.task"
OUTPUT_JSON = None   # resolved at runtime: <OUTPUT_DIR>/all_data.json

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


# ── 1. Parse IMU packets (40 bytes each, Azure Kinect format) ─────────────────
# Format: acc_timestamp_usec (uint64) + acc_xyz (3×float32) + padding (uint32)
#       + gyro_timestamp_usec (uint64) + gyro_xyz (3×float32) + padding (uint32)
IMU_STRUCT = struct.Struct('<Q 3f Q 3f')   # 8+12+8+12 = 40 bytes, no padding

def parse_imu(container):
    """Extract all IMU samples. Returns list of dicts."""
    imu_records = []
    sub_stream = container.streams[3]
    for pkt in container.demux([sub_stream]):
        if pkt.size == 40:
            vals = IMU_STRUCT.unpack(bytes(pkt))
            imu_records.append({
                "acc_timestamp_usec": vals[0],
                "acc_x_ms2":   round(vals[1], 6),
                "acc_y_ms2":   round(vals[2], 6),
                "acc_z_ms2":   round(vals[3], 6),
                "gyro_timestamp_usec": vals[4],
                "gyro_x_rads": round(vals[5], 6),
                "gyro_y_rads": round(vals[6], 6),
                "gyro_z_rads": round(vals[7], 6),
            })
    return imu_records


# ── 2. Re-run MediaPipe with pose_world_landmarks (3D in meters) ───────────────
def extract_world_poses(cap, fps, total, model_path):
    """
    Run MediaPipe and collect BOTH normalized (image) landmarks AND
    world landmarks (3D in meters, hip-centered).
    Returns dict: frame_idx -> {"image": [...], "world_3d": [...]}
    """
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    options   = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=1,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    frame_data = {}
    frame_idx  = 0

    with tqdm(total=total, desc="Re-extracting pose + 3D world coords") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_idx / fps * 1000)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect_for_video(mp_image, timestamp_ms)

            entry = {"detected": False, "timestamp_sec": round(frame_idx / fps, 4)}

            if result.pose_landmarks:
                lms_img   = result.pose_landmarks[0]
                lms_world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None

                image_lm = []
                for i, name in enumerate(LANDMARK_NAMES):
                    lm = lms_img[i]
                    image_lm.append({
                        "name": name,
                        "x": round(lm.x, 5),   # normalized image coord
                        "y": round(lm.y, 5),
                        "z": round(lm.z, 5),   # relative depth (normalized)
                        "visibility": round(lm.visibility, 4),
                    })

                world_lm = []
                if lms_world:
                    for i, name in enumerate(LANDMARK_NAMES):
                        wlm = lms_world[i]
                        world_lm.append({
                            "name": name,
                            "x_m": round(wlm.x, 5),   # meters, hip-centered
                            "y_m": round(wlm.y, 5),
                            "z_m": round(wlm.z, 5),
                            "visibility": round(wlm.visibility, 4),
                        })

                entry.update({
                    "detected":  True,
                    "image_landmarks": image_lm,   # normalized 0-1 coords
                    "world_landmarks": world_lm,   # real 3D in meters
                })

            frame_data[frame_idx] = entry
            frame_idx += 1
            pbar.update(1)

    landmarker.close()
    return frame_data


# ── 3. Load depth stats CSV ───────────────────────────────────────────────────
def load_depth_stats(path):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path).set_index("frame")
    return df.to_dict(orient="index")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge pose + depth + IMU data into a single all_data.json.")
    parser.add_argument("input", nargs="?", default=INPUT_FILE,
                        help="Input .mkv file (default: %(default)s)")
    parser.add_argument("-o", "--output", default=OUTPUT_DIR,
                        help="Output directory (must already contain depth_stats.csv to include depth)")
    parser.add_argument("--model", default=MODEL_PATH,
                        help="Path to MediaPipe pose_landmarker_full.task (auto-downloaded if missing)")
    args = parser.parse_args()

    INPUT_FILE  = args.input
    OUTPUT_DIR  = args.output
    MODEL_PATH  = args.model
    OUTPUT_JSON = os.path.join(OUTPUT_DIR, "all_data.json")

    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] {INPUT_FILE} not found"); sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Downloading pose model (~25 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[INFO] Saved to: {MODEL_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Video metadata
    cap   = cv2.VideoCapture(INPUT_FILE)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[1/3] Extracting pose + 3D world landmarks ...")
    frame_data = extract_world_poses(cap, fps, total, MODEL_PATH)
    cap.release()

    print(f"\n[2/3] Extracting IMU data ...")
    container  = av.open(INPUT_FILE)
    imu_data   = parse_imu(container)
    container.close()
    print(f"      IMU samples: {len(imu_data)}")

    print(f"\n[3/3] Loading depth stats ...")
    depth_map = load_depth_stats(os.path.join(OUTPUT_DIR, "depth_stats.csv"))

    # Assemble final JSON
    doc = {
        "metadata": {
            "source_file": INPUT_FILE,
            "color_resolution": f"{w}x{h}",
            "depth_resolution": "640x576",
            "fps": fps,
            "total_frames": total,
            "duration_sec": round(total / fps, 2),
            "streams": {
                "0": "color MJPEG 1280x720",
                "1": "depth gray16be 640x576 (mm)",
                "2": "IR gray16be 640x576",
                "3": "IMU (acc m/s², gyro rad/s ~1kHz)",
                "4": "calibration attachment",
            },
            "landmark_coordinate_systems": {
                "image_landmarks_x_y": "normalized 0.0-1.0, origin=top-left",
                "image_landmarks_z": "relative depth, normalized, hip-centered, negative=closer to camera",
                "world_landmarks_x_y_z": "meters, hip-centered (origin = midpoint of hips), y-axis up",
                "visibility": "confidence 0.0-1.0",
            },
        },
        "frames": [],
        "imu": imu_data,
    }

    for idx in range(total):
        pose    = frame_data.get(idx, {"detected": False, "timestamp_sec": round(idx / fps, 4)})
        depth   = depth_map.get(idx, {})

        frame_entry = {
            "frame":         idx,
            "timestamp_sec": pose["timestamp_sec"],
            "pose": {
                "detected":        pose.get("detected", False),
                "image_landmarks": pose.get("image_landmarks", []),
                "world_landmarks": pose.get("world_landmarks", []),
            },
            "depth": {
                "valid_ratio":   depth.get("valid_ratio"),
                "mean_mm":       depth.get("depth_mean_mm"),
                "std_mm":        depth.get("depth_std_mm"),
                "min_mm":        depth.get("depth_min_mm"),
                "max_mm":        depth.get("depth_max_mm"),
                "p25_mm":        depth.get("depth_p25_mm"),
                "p75_mm":        depth.get("depth_p75_mm"),
            } if depth else None,
        }
        doc["frames"].append(frame_entry)

    print(f"\nWriting JSON to {OUTPUT_JSON} ...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(doc, f, indent=2, allow_nan=True)

    size_mb = os.path.getsize(OUTPUT_JSON) / 1e6
    print(f"[Done] {OUTPUT_JSON}  ({size_mb:.1f} MB)")
    print(f"  - {total} frames")
    print(f"  - {len(imu_data)} IMU samples")
    print(f"  - Each frame: pose (image + 3D world) + depth stats")
