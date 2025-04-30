import cv2
import numpy as np

# --- Parameters ---
lk_params = dict(
    winSize  = (21, 21),
    maxLevel = 3,
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01
    )
)
feature_params = dict(
    maxCorners    = 100,
    qualityLevel  = 0.01,
    minDistance   = 8,
    blockSize     = 7
)
min_features = 10  # when to re-detect

# --- Open webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# --- Read first frame & let user select ROI ---
ret, first_frame = cap.read()
if not ret:
    raise IOError("Cannot read from webcam")
roi = cv2.selectROI("Select object to track", first_frame, False, False)
cv2.destroyWindow("Select object to track")

(x, y, w, h) = tuple(map(int, roi))
old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

def detect_features(gray, rect):
    """ Detect good features inside rect = (x,y,w,h) """
    x, y, w, h = rect
    mask = np.zeros_like(gray)
    mask[y:y+h, x:x+w] = 255
    pts = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
    return None if pts is None else pts

# --- Initialize feature points inside ROI ---
old_pts = detect_features(old_gray, (x,y,w,h))
if old_pts is None:
    raise RuntimeError("No features found in the selected ROI.")

# create image for drawing tracks
mask = np.zeros_like(first_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam disconnected")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    new_pts, status, err = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, old_pts, None, **lk_params
    )
    if new_pts is None:
        status = np.zeros((old_pts.shape[0],), dtype=bool)
    else:
        status = status.reshape(-1).astype(bool)

    good_new = new_pts[status] if new_pts is not None else np.empty((0,1,2))
    good_old = old_pts[status]

    # if too few points, re-detect inside current object region
    if len(good_new) < min_features:
        # Estimate new bounding box from last known good points
        if len(good_new) > 0:
            pts_xy = good_new.reshape(-1,2)
            x, y, w, h = cv2.boundingRect(np.float32(pts_xy))
        # Detect fresh points
        old_pts = detect_features(frame_gray, (x,y,w,h))
        mask[:] = 0
        old_gray = frame_gray.copy()
        continue

    # draw tracking lines & points
    for (new, old) in zip(good_new.reshape(-1,2), good_old.reshape(-1,2)):
        a, b = map(int, new)
        c, d = map(int, old)
        mask = cv2.line(mask, (a,b), (c,d), (0,255,0), 2)
        frame = cv2.circle(frame, (a,b), 4, (0,0,255), -1)

    # draw updated bounding box
    pts_xy = good_new.reshape(-1,2)
    bx, by, bw, bh = cv2.boundingRect(np.float32(pts_xy))
    cv2.rectangle(frame, (bx,by), (bx+bw, by+bh), (255,0,0), 2)

    # overlay and show
    output = cv2.add(frame, mask)
    cv2.imshow('Real-time LK Object Tracker', output)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break

    # update for next frame
    old_gray = frame_gray.copy()
    old_pts  = good_new.reshape(-1,1,2)

cap.release()
cv2.destroyAllWindows()
