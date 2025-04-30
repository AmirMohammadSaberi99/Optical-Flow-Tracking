# Optical Flow Tracking with Lucas–Kanade

This repository contains two Python scripts demonstrating the use of OpenCV’s Lucas–Kanade optical flow for real-time and dynamic feature-based tracking:

- **`real_time_object_tracker.py`**: Select an object via ROI (Region of Interest) and track Shi–Tomasi corners within that box in real time, visualizing motion vectors with arrows and an updated bounding box.
- **`dynamic_feature_lk_flow.py`**: Automatically detect strong Shi–Tomasi corners on each frame, track them with Lucas–Kanade optical flow, and re-detect new features dynamically when existing tracks are lost or sparse.

---

## Requirements

- Python 3.6 or higher
- [OpenCV Python](https://pypi.org/project/opencv-python/)

Install dependencies:

```bash
pip install opencv-python
```

---

## Repository Structure

```
├── real_time_object_tracker.py  # Select ROI + track features in real time
├── dynamic_feature_lk_flow.py   # Auto-select + track features dynamically
└── README.md                    # This file
```

---

## Usage

### 1. Real-Time Object Tracker

```bash
python real_time_object_tracker.py
```

1. A window will open showing the webcam feed.
2. Draw a bounding box around the object you want to track, then press **Enter** or **Space** to confirm.
3. Watch as the script tracks feature points inside the box, drawing green arrows for flow vectors, red dots for point positions, and a blue rectangle around the object.
4. Press **ESC** at any time to exit.

### 2. Dynamic Feature LK Flow

```bash
python dynamic_feature_lk_flow.py
```

1. This script opens the webcam and immediately starts detecting strong corner features.
2. Tracks those points with Lucas–Kanade optical flow, drawing arrows for motion.
3. If feature count falls below a threshold or every few frames, it re-detects new corners and continues.
4. Press **ESC** to stop.

---

## Script Details

### Common Parameters

Both scripts use these default parameters, which you can tweak at the top of each file:

```python
# Parameters for Shi–Tomasi corner detection
default_feature_params = dict(
    maxCorners    = 100,       # maximum number of corners to detect
    qualityLevel  = 0.01,      # minimum accepted quality of corners
    minDistance   = 8,         # minimum distance between corners
    blockSize     = 7          # size of the averaging block
)

# Lucas–Kanade optical flow parameters
default_lk_params = dict(
    winSize  = (21, 21),       # search window size
    maxLevel = 3,              # number of pyramid levels
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01
    )
)
```

### real_time_object_tracker.py

- **ROI selection**: Uses `cv2.selectROI` on the first frame to let you draw a box around your target.
- **Feature detection**: Runs `cv2.goodFeaturesToTrack` inside the ROI to get initial Shi–Tomasi corners.
- **Tracking loop**:
  1. Computes flow with `cv2.calcOpticalFlowPyrLK`.
  2. Draws arrowed lines (`cv2.arrowedLine`) and circles for each tracked point.
  3. Re-detects features inside the updated bounding box whenever the count falls below a minimum.
  4. Displays in real time until **ESC** is pressed.

### dynamic_feature_lk_flow.py

- **Initial detection**: Runs `goodFeaturesToTrack` on the entire first frame.
- **Dynamic loop**:
  1. Tracks points with `calcOpticalFlowPyrLK`.
  2. Filters out lost points (`status==0`).
  3. Draws arrows and circles for surviving tracks.
  4. Every few frames or if count is low, masks out existing points and re-runs `goodFeaturesToTrack` to inject new strong features.
  5. Clears previous trajectories on re-detection to keep visualization clean.

---

## License

This project is released under the MIT License. Feel free to use and modify!

