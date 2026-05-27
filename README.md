# kinect-extractor

Extract human-motion data from **Azure Kinect MKV** recordings into CSV / JSON
for downstream analysis. No Azure Kinect SDK required — only OpenCV, PyAV, and
MediaPipe.

Given an `.mkv` recorded by `k4arecorder`, the scripts pull out:

- **2D pose** (33 MediaPipe BlazePose landmarks) per frame, per person
- **3D world pose** in meters (hip-centered)
- **Depth statistics** per frame (mean, std, percentiles, valid ratio)
- **IMU** samples (accelerometer + gyroscope, ~1 kHz)
- Visualization videos (pose overlay, colormap depth) and time-series plots

## What gets produced

```
output/
├── joint_positions.csv               # single-person pose, one row per frame
├── multiperson_joints.csv            # multi-person pose with person_id (+ side when n=2)
├── person_0_left_joints.csv          # per-person split (left/right when n=2)
├── person_1_right_joints.csv         #   …or person_N_joints.csv otherwise
├── depth_stats.csv                   # per-frame depth statistics
├── all_data.json                     # everything merged (pose + depth + IMU)
├── pose_overlay.mp4                  # skeleton drawn over color video
├── multiperson_pose.mp4              # color-coded per-person skeletons w/ side labels
├── depth_colormap.mp4                # false-color depth video
└── *.png                             # overview plots
```

See [`examples/`](examples/) for what the CSVs and plots look like.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended.

The MediaPipe pose model (`pose_landmarker_full.task`, ~25 MB) is downloaded
automatically on first run and cached in the working directory.

> **MediaPipe version note.** `requirements.txt` pins `mediapipe<0.10.30`.
> Versions ≥ 0.10.30 ship a new C-bindings layer that requires the system
> library `libGLESv2.so.2` (Debian/Ubuntu: `sudo apt install libgles2`). If
> you have root on your machine, installing that one package lets you use
> any recent MediaPipe. If you don't have sudo, stick with the pinned
> version — it only needs EGL, which most Linux desktops already have.

## Usage

All four scripts accept the input `.mkv` as a positional argument and `-o` for
the output directory. Defaults to `input.mkv` and `./output`.

### 1. Single-person 2D pose → CSV

```bash
python extract_kinect.py path/to/recording.mkv -o output/
```

Produces `joint_positions.csv` with columns:

```
frame, timestamp_sec, detected,
nose_x, nose_y, nose_z, nose_vis,        # 33 landmarks × 4 columns each
left_eye_inner_x, ..., right_foot_index_vis
```

`x`, `y` are normalized image coordinates (`0.0`–`1.0`, origin top-left).
`z` is normalized relative depth (hip-centered, negative = closer).
`vis` is detection confidence `0.0`–`1.0`.

### 2. Depth statistics → CSV

```bash
python extract_depth.py path/to/recording.mkv -o output/
```

Produces `depth_stats.csv`:

```
frame, timestamp_sec, valid_ratio,
depth_mean_mm, depth_std_mm, depth_min_mm, depth_max_mm,
depth_p25_mm, depth_p75_mm
```

`--save-npy` will additionally dump each raw uint16 depth frame as `.npy`
(warning: tens of GB for long recordings).

### 3. Multi-person tracked pose → CSV

```bash
python extract_multiperson.py path/to/recording.mkv -o output/ --n-persons 3
```

Uses Hungarian matching on key-joint positions **plus** a torso HSV color
histogram (EMA-updated) so person IDs stay stable across frames even with
brief occlusions. Produces a combined `multiperson_joints.csv` and one
`person_N_joints.csv` per tracked person.

**Left/right relabeling.** After the detection pass, each person's mean
`nose_x` over the whole recording is computed and IDs are reassigned so that
**`person_id=0` is always the leftmost person, `person_id=1` the next, etc.**
This makes results reproducible across runs and across recordings.

When `--n-persons 2`, the CSV also gets a `side` column (`"left"` /
`"right"`) and per-person files are named `person_0_left_joints.csv`,
`person_1_right_joints.csv`. The overlay video labels each skeleton
`P0(left)` / `P1(right)` accordingly — overlay rendering happens in a
second pass after the relabel, so labels and CSV always agree.

**Example: two people, one on each side**

```bash
python extract_multiperson.py study1_output_1.mkv -o sample_run/ --n-persons 2
```

```
[OK] Combined CSV (1200 rows): sample_run/multiperson_joints.csv
     Person 0 (left): 600 frames (100% presence) → sample_run/person_0_left_joints.csv
     Person 1 (right): 600 frames (100% presence) → sample_run/person_1_right_joints.csv
```

### 4. Merge everything into one JSON

```bash
python build_json.py path/to/recording.mkv -o output/
```

Re-runs pose extraction to also collect 3D **world landmarks** (in meters,
hip-centered), parses the IMU stream (Stream 3 of the MKV), and joins them
with `depth_stats.csv` if present. Output: `all_data.json`.

Run this **after** `extract_depth.py` if you want depth fields populated.

## Pipeline for a new recording

```bash
# 1. quick check (single-person pose)
python extract_kinect.py session.mkv -o session_out/

# 2. depth statistics
python extract_depth.py session.mkv -o session_out/

# 3. multi-person tracking (if scene has >1 person)
python extract_multiperson.py session.mkv -o session_out/ --n-persons 3

# 4. combined JSON for downstream tools
python build_json.py session.mkv -o session_out/
```

For a quick test, add `--max-frames 300` to any script to limit processing
to the first 300 frames.

## MKV stream layout (Azure Kinect)

`k4arecorder` writes MKV files with this stream layout:

| Stream | Type  | Format     | Resolution | Notes                           |
| ------ | ----- | ---------- | ---------- | ------------------------------- |
| 0      | Color | MJPEG      | 1280×720   | read via OpenCV                 |
| 1      | Depth | gray16be   | 640×576    | uint16, values in mm            |
| 2      | IR    | gray16be   | 640×576    |                                 |
| 3      | IMU   | raw 40 B   | ~1 kHz     | acc m/s² + gyro rad/s (Struct)  |
| 4      | meta  | attachment | —          | calibration JSON                |

If your recording uses a different layout, pass a different `--depth-stream` /
edit the constants near the top of each script.

## Common knobs

All scripts share these flags (where applicable):

| Flag                  | Effect                                              |
| --------------------- | --------------------------------------------------- |
| `--skip-frames N`     | Process every Nth frame (1 = all)                   |
| `--max-frames N`      | Stop after N frames (useful for testing)            |
| `--no-video`          | Skip writing the overlay/colormap MP4               |
| `-o, --output DIR`    | Output directory                                    |

`extract_multiperson.py` additionally exposes `--n-persons` and `--alpha`
(position vs. color cost weight, 0..1).

## Landmark reference

MediaPipe BlazePose, 33 points in this order:

```
0  nose                    11 left_shoulder           22 right_thumb
1  left_eye_inner          12 right_shoulder          23 left_hip
2  left_eye                13 left_elbow              24 right_hip
3  left_eye_outer          14 right_elbow             25 left_knee
4  right_eye_inner         15 left_wrist              26 right_knee
5  right_eye               16 right_wrist             27 left_ankle
6  right_eye_outer         17 left_pinky              28 right_ankle
7  left_ear                18 right_pinky             29 left_heel
8  right_ear               19 left_index              30 right_heel
9  mouth_left              20 right_index             31 left_foot_index
10 mouth_right             21 left_thumb              32 right_foot_index
```

See `examples/landmark_diagram.png` for a labeled skeleton.

## Troubleshooting

- **`Cannot open file`** — OpenCV can't decode the MJPEG stream. Make sure
  FFmpeg is installed (`apt install ffmpeg` or the FFmpeg wheels bundled
  with `opencv-python` should suffice on most systems).
- **`libGLESv2.so.2: cannot open shared object file`** — your MediaPipe is
  ≥ 0.10.30. Either `sudo apt install libgles2` or downgrade:
  `pip install 'mediapipe<0.10.30'`.
- **`mediapipe` install fails on Linux** — install Python 3.10–3.12; MediaPipe
  doesn't ship wheels for 3.13 yet.
- **Empty IMU samples in `all_data.json`** — Stream 3 may not be IMU in your
  recording. Open the file with `ffprobe` and adjust the index in `build_json.py`.
- **Person IDs swap between frames** — try increasing `--alpha` toward 1.0
  (rely more on position) or decreasing toward 0.0 (rely more on color).

## License

These scripts use MediaPipe (Apache 2.0) and OpenCV (Apache 2.0). The code
in this repo is provided as-is for research use.
