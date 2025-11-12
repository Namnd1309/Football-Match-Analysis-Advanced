import pickle
import cv2
import numpy as np
import sys
sys.path.append("../")
from utils import measure_distance, measure_xy_distance
import os

class CameraMovementEstimator:
    def __init__(self, frame):
        self.minimum_distance = 5
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = first_frame_gray.shape
        mask_features = np.zeros_like(first_frame_gray)

        # ✅ Cập nhật an toàn — chỉ lấy 5% biên trái/phải ảnh
        left_w = max(1, int(w * 0.05))
        right_start = max(0, w - left_w)
        mask_features[:, :left_w] = 1
        mask_features[:, right_start:] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # ✅ Đọc từ stub nếu có
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"[INFO] Loading camera movement from {stub_path}")
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        print("[INFO] Calculating camera movement...")
        camera_movement = [[0, 0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        if old_features is None:
            print("[WARN] No features found in the first frame.")
            return camera_movement

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            if old_features is None or len(old_features) == 0:
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                continue

            new_features, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            if new_features is None:
                continue

            max_distance = 0
            cam_x, cam_y = 0, 0
            for new, old in zip(new_features, old_features):
                new_pt, old_pt = new.ravel(), old.ravel()
                dist = measure_distance(new_pt, old_pt)
                if dist > max_distance:
                    max_distance = dist
                    cam_x, cam_y = measure_xy_distance(old_pt, new_pt)

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [cam_x, cam_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)
            print(f"[INFO] Saved camera movement to {stub_path}")

        return camera_movement

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    pos = track_info["position"]
                    cam_move = camera_movement_per_frame[frame_num]
                    adjusted = (pos[0] - cam_move[0], pos[1] - cam_move[1])
                    tracks[object_type][frame_num][track_id]["position_adjusted"] = adjusted