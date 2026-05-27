"""
Azure Kinect MKV Depth Map Extraction
Stream layout:
  Stream 0  - Color  MJPEG  1280x720
  Stream 1  - Depth  raw    640x576  gray16be  (values in mm)
  Stream 2  - IR     raw    640x576  gray16be

This script:
  1. Reads depth frames via PyAV (no Azure Kinect SDK required)
  2. Saves per-frame depth arrays as .npy (optional)
  3. Saves a colormap visualization video
  4. Generates depth statistics CSV (mean/std/min/max/valid ratio per frame)
  5. Plots depth distribution over time
"""

import argparse
import av
import numpy as np
import pandas as pd
import cv2
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─── Defaults (overridable via CLI) ───────────────────────────────────────────
INPUT_FILE      = "input.mkv"
OUTPUT_DIR      = "output"
DEPTH_STREAM    = 1          # MKV stream index for depth (0=color, 1=depth, 2=IR)
IR_STREAM       = 2          # MKV stream index for IR
SAVE_NPY        = False      # True = save each frame as .npy (large disk usage)
SAVE_DEPTH_VID  = True       # Save false-color depth visualization video
SAVE_IR_VID     = False      # Save IR video
SKIP_FRAMES     = 1          # Process every Nth frame
MAX_FRAMES      = None       # None = all frames
DEPTH_MIN_MM    = 300        # Ignore depth below this (noise / too close)
DEPTH_MAX_MM    = 5000       # Ignore depth above this (out of range)
COLORMAP        = cv2.COLORMAP_JET   # Depth visualization colormap
# ─────────────────────────────────────────────────────────────────────────────


def open_container(path):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    container = av.open(path)
    for i, s in enumerate(container.streams):
        tag = f"{s.type}, {getattr(s, 'width','?')}x{getattr(s, 'height','?')}"
        print(f"  Stream {i}: {tag}")
    return container


def decode_gray16be(frame_av) -> np.ndarray:
    """Convert a PyAV gray16be frame to a uint16 NumPy array (H, W)."""
    arr = np.frombuffer(bytes(frame_av.planes[0]), dtype=np.uint16)
    # gray16be is big-endian on disk; PyAV may or may not byte-swap for us
    arr = arr.byteswap().view(arr.dtype.newbyteorder())  # ensure native little-endian (NumPy 2.0 compat)
    return arr.reshape(frame_av.height, frame_av.width)


def depth_to_colormap(depth_mm: np.ndarray, d_min=DEPTH_MIN_MM, d_max=DEPTH_MAX_MM) -> np.ndarray:
    """
    Map depth values (mm) to an 8-bit BGR colormap image.
    Pixels outside [d_min, d_max] are shown as black.
    """
    valid = (depth_mm >= d_min) & (depth_mm <= d_max)
    norm  = np.zeros_like(depth_mm, dtype=np.float32)
    norm[valid] = (depth_mm[valid] - d_min) / (d_max - d_min)
    norm_u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm_u8, COLORMAP)
    colored[~valid] = 0   # black for invalid pixels
    return colored


def extract_depth(container, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if SAVE_NPY:
        npy_dir = os.path.join(out_dir, "depth_npy")
        os.makedirs(npy_dir, exist_ok=True)

    # Setup video writers (will init on first frame)
    depth_writer = None
    ir_writer    = None

    records     = []
    frame_idx   = 0
    processed   = 0
    limit       = MAX_FRAMES if MAX_FRAMES else float("inf")

    # Demux only the streams we need
    wanted = {DEPTH_STREAM}
    if SAVE_IR_VID:
        wanted.add(IR_STREAM)

    stream_objs = [s for i, s in enumerate(container.streams) if i in wanted]

    with tqdm(desc="Extracting depth", unit="frames") as pbar:
        for packet in container.demux(stream_objs):
            if processed >= limit:
                break

            stream_idx = packet.stream_index

            for av_frame in packet.decode():
                if av_frame is None:
                    continue

                # Compute timestamp in seconds
                if av_frame.pts is not None and av_frame.time_base is not None:
                    ts_sec = float(av_frame.pts * av_frame.time_base)
                else:
                    ts_sec = float(frame_idx) / 15.0   # fallback: assume 15 fps

                if stream_idx == DEPTH_STREAM:
                    if frame_idx % SKIP_FRAMES != 0:
                        frame_idx += 1
                        continue

                    depth_mm = decode_gray16be(av_frame)

                    # Statistics (valid pixels only)
                    valid_mask = (depth_mm >= DEPTH_MIN_MM) & (depth_mm <= DEPTH_MAX_MM)
                    valid_vals = depth_mm[valid_mask]
                    row = {
                        "frame":          frame_idx,
                        "timestamp_sec":  round(ts_sec, 4),
                        "valid_ratio":    round(valid_mask.mean(), 4),
                        "depth_mean_mm":  round(float(valid_vals.mean()), 1) if valid_vals.size else np.nan,
                        "depth_std_mm":   round(float(valid_vals.std()),  1) if valid_vals.size else np.nan,
                        "depth_min_mm":   int(valid_vals.min())            if valid_vals.size else np.nan,
                        "depth_max_mm":   int(valid_vals.max())            if valid_vals.size else np.nan,
                        "depth_p25_mm":   round(float(np.percentile(valid_vals, 25)), 1) if valid_vals.size else np.nan,
                        "depth_p75_mm":   round(float(np.percentile(valid_vals, 75)), 1) if valid_vals.size else np.nan,
                    }
                    records.append(row)

                    # Optional: save raw depth as .npy
                    if SAVE_NPY:
                        np.save(os.path.join(npy_dir, f"depth_{frame_idx:05d}.npy"), depth_mm)

                    # Colorized depth image
                    colored = depth_to_colormap(depth_mm)

                    if SAVE_DEPTH_VID:
                        if depth_writer is None:
                            h, w = colored.shape[:2]
                            vpath = os.path.join(out_dir, "depth_colormap.mp4")
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            depth_writer = cv2.VideoWriter(vpath, fourcc, 15.0 / max(SKIP_FRAMES, 1), (w, h))
                            print(f"\n[INFO] Depth video: {vpath}")
                        depth_writer.write(colored)

                    frame_idx += 1
                    processed += 1
                    pbar.update(1)

                elif stream_idx == IR_STREAM and SAVE_IR_VID:
                    ir_raw = decode_gray16be(av_frame)
                    # Normalize IR to 8-bit for visualization
                    ir_8 = cv2.normalize(ir_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    ir_bgr = cv2.cvtColor(ir_8, cv2.COLOR_GRAY2BGR)
                    if ir_writer is None:
                        h, w = ir_bgr.shape[:2]
                        ipath = os.path.join(out_dir, "ir_video.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        ir_writer = cv2.VideoWriter(ipath, fourcc, 15.0, (w, h))
                        print(f"\n[INFO] IR video: {ipath}")
                    ir_writer.write(ir_bgr)

    if depth_writer:
        depth_writer.release()
    if ir_writer:
        ir_writer.release()

    return records


def save_depth_results(records, out_dir):
    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "depth_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Depth stats saved: {csv_path}  ({len(df)} frames)")

    # ── Plot: depth over time ─────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df["timestamp_sec"], df["depth_mean_mm"],  label="Mean",  lw=0.8, color="steelblue")
    axes[0].fill_between(
        df["timestamp_sec"],
        df["depth_p25_mm"], df["depth_p75_mm"],
        alpha=0.25, color="steelblue", label="IQR (25-75%)"
    )
    axes[0].set_title("Mean Depth Over Time (mm)")
    axes[0].set_ylabel("Depth (mm)")
    axes[0].legend(fontsize=8)

    axes[1].plot(df["timestamp_sec"], df["valid_ratio"], color="seagreen", lw=0.8)
    axes[1].set_title("Valid Pixel Ratio (depth within range)")
    axes[1].set_ylabel("Ratio")
    axes[1].set_ylim(0, 1)

    axes[2].plot(df["timestamp_sec"], df["depth_std_mm"], color="coral", lw=0.8)
    axes[2].set_title("Depth Std Dev Over Time (scene complexity / motion)")
    axes[2].set_xlabel("Time (sec)")
    axes[2].set_ylabel("Std (mm)")

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "depth_overview.png")
    plt.savefig(plot_path, dpi=150)
    print(f"[OK] Depth plot saved: {plot_path}")
    plt.close()

    return df


def save_sample_frames(container, out_dir, n_samples=6):
    """Save N evenly-spaced depth frames as colormap PNG images for quick review."""
    os.makedirs(out_dir, exist_ok=True)
    depth_stream = container.streams[DEPTH_STREAM]
    all_frames   = []

    for packet in container.demux([depth_stream]):
        for av_frame in packet.decode():
            if av_frame is not None:
                all_frames.append(decode_gray16be(av_frame))
            if len(all_frames) >= 500:   # sample from first 500 frames only
                break
        if len(all_frames) >= 500:
            break

    if not all_frames:
        return

    indices = np.linspace(0, len(all_frames)-1, n_samples, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, idx in zip(axes.flat, indices):
        d = all_frames[idx]
        colored = depth_to_colormap(d)
        ax.imshow(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Frame {idx}  (mean {d[d>0].mean():.0f} mm)")
        ax.axis("off")
    plt.suptitle("Sample Depth Frames (false color: blue=near, red=far)")
    plt.tight_layout()
    sample_path = os.path.join(out_dir, "depth_sample_frames.png")
    plt.savefig(sample_path, dpi=150)
    print(f"[OK] Sample frames saved: {sample_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract depth-frame statistics + false-color depth video from a Kinect MKV.")
    parser.add_argument("input", nargs="?", default=INPUT_FILE,
                        help="Input .mkv file (default: %(default)s)")
    parser.add_argument("-o", "--output", default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--skip-frames", type=int, default=SKIP_FRAMES,
                        help="Process every Nth frame (default: %(default)d)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit total frames processed (default: all)")
    parser.add_argument("--save-npy", action="store_true",
                        help="Save raw uint16 depth frames as .npy (large disk usage)")
    parser.add_argument("--no-depth-video", action="store_true",
                        help="Skip writing the colormap depth video")
    parser.add_argument("--save-ir-video", action="store_true",
                        help="Also write the IR stream as an .mp4")
    parser.add_argument("--depth-min-mm", type=int, default=DEPTH_MIN_MM,
                        help="Lower depth clip (mm, default: %(default)d)")
    parser.add_argument("--depth-max-mm", type=int, default=DEPTH_MAX_MM,
                        help="Upper depth clip (mm, default: %(default)d)")
    args = parser.parse_args()

    INPUT_FILE     = args.input
    OUTPUT_DIR     = args.output
    SKIP_FRAMES    = args.skip_frames
    MAX_FRAMES     = args.max_frames
    SAVE_NPY       = args.save_npy
    SAVE_DEPTH_VID = not args.no_depth_video
    SAVE_IR_VID    = args.save_ir_video
    DEPTH_MIN_MM   = args.depth_min_mm
    DEPTH_MAX_MM   = args.depth_max_mm

    print(f"[INFO] Opening: {INPUT_FILE}")
    container = open_container(INPUT_FILE)

    print("\n--- Phase 1: Extract depth statistics and video ---")
    records = extract_depth(container, OUTPUT_DIR)
    container.close()

    df = save_depth_results(records, OUTPUT_DIR)

    print("\n--- Phase 2: Save sample depth frame images ---")
    container2 = open_container(INPUT_FILE)
    save_sample_frames(container2, OUTPUT_DIR)
    container2.close()

    print(f"\n[Done] All depth outputs in ./{OUTPUT_DIR}/")
    print(f"  depth_stats.csv          — per-frame depth statistics")
    print(f"  depth_colormap.mp4       — false-color depth video")
    print(f"  depth_overview.png       — depth time series plots")
    print(f"  depth_sample_frames.png  — 6 sample depth frames")
