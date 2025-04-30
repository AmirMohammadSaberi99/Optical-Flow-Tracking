import cv2
import numpy as np

# --- Parameters for Lucas-Kanade optical flow ---
lk_params = dict(
    winSize  = (15, 15),
    maxLevel = 2,
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

# --- Parameters for Shi-Tomasi corner detection ---
feature_params = dict(
    maxCorners    = 200,
    qualityLevel  = 0.01,
    minDistance   = 7,
    blockSize     = 7
)

# Minimum number of tracked points before re-detecting
min_points = 50
# How often (in frames) to try detecting new points even if we still have enough
detect_interval = 5

def main():
    cap = cv2.VideoCapture(0)  # use 0 for webcam or replace with filename
    if not cap.isOpened():
        raise IOError("Cannot open video source")

    # Read first frame
    ret, old_frame = cap.read()
    if not ret:
        raise IOError("Cannot read first frame")
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Initial feature detection
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    if p0 is None:
        raise RuntimeError("No features found in the first frame")

    # Create a mask image for drawing
    mask = np.zeros_like(old_frame)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video / stream.")
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow to track points
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        if p1 is None:
            # nothing tracked — re-detect from scratch
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            mask[:] = 0
            old_gray = frame_gray.copy()
            frame_idx += 1
            continue

        # select good points
        st = st.reshape(-1)
        good_new = p1[st == 1].reshape(-1, 2)
        good_old = p0.reshape(-1, 2)[st == 1]

        # draw the flow vectors as arrows
        for new, old in zip(good_new, good_old):
            a, b = new.astype(int)
            c, d = old.astype(int)
            mask = cv2.arrowedLine(mask, (c, d), (a, b), (0, 255, 0), 2, tipLength=0.3)
            frame = cv2.circle(frame, (a, b), 3, (0, 0, 255), -1)

        # overlay the arrows on the frame
        img = cv2.add(frame, mask)
        cv2.imshow('Dynamic Feature LK Flow', img)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:  # ESC to exit
            break

        # if too few points, or every detect_interval frames, re-detect
        if len(good_new) < min_points or frame_idx % detect_interval == 0:
            # build a mask to avoid re-detecting on existing points
            detect_mask = np.ones_like(frame_gray, dtype=np.uint8) * 255
            for x, y in good_new.astype(int):
                cv2.circle(detect_mask, (x, y), feature_params['minDistance'], 0, -1)
            new_pts = cv2.goodFeaturesToTrack(frame_gray, mask=detect_mask, **feature_params)
            if new_pts is not None:
                # append new points to surviving ones
                good_new = np.vstack([good_new, new_pts.reshape(-1, 2)])
            mask[:] = 0  # clear trajectories after redetection

        # prepare for next frame
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
