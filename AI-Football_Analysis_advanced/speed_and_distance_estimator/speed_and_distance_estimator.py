import cv2
import sys
sys.path.append("../")
from utils import measure_distance, get_foot_position


class SpeedAndDistanceEstimator:
    def __init__(self):
        self.frame_window = 5     # tính vận tốc mỗi 5 frame
        self.frame_rate = 24      # video 24 fps

    def add_speed_and_distance_to_track(self, tracks):
        total_distance = {}

        for object_name, object_tracks in tracks.items():
            # --- ⚙️ Nếu là bóng thì vẫn tính ---
            if object_name == "referees":
                continue

            number_of_frames = len(object_tracks)

            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id].get("position_transformed")
                    end_position = object_tracks[last_frame][track_id].get("position_transformed")

                    if start_position is None or end_position is None:
                        continue

                    # --- Tính quãng đường và tốc độ ---
                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate

                    if time_elapsed == 0:
                        continue

                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    # --- Cộng dồn quãng đường ---
                    total_distance.setdefault(object_name, {})
                    total_distance[object_name].setdefault(track_id, 0)
                    total_distance[object_name][track_id] += distance_covered

                    # --- Gán tốc độ + quãng đường cho từng frame trong khoảng ---
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object_name][frame_num_batch]:
                            continue
                        tracks[object_name][frame_num_batch][track_id]["speed"] = speed_km_per_hour
                        tracks[object_name][frame_num_batch][track_id]["distance"] = total_distance[object_name][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object_name, object_tracks in tracks.items():
                if object_name == "referees":
                    continue

                for _, track_info in object_tracks[frame_num].items():
                    speed = track_info.get("speed")
                    distance = track_info.get("distance")

                    if speed is None or distance is None:
                        continue

                    # --- Nếu là bóng thì hiển thị text phía trên ---
                    if object_name == "ball":
                        bbox = track_info["bbox"]
                        cx, cy = get_foot_position(bbox)
                        cv2.putText(frame, f"Ball {speed:.2f} km/h",
                                    (int(cx), int(cy) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.putText(frame, f"{distance:.2f} m",
                                    (int(cx), int(cy)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        # --- Cầu thủ ---
                        bbox = track_info["bbox"]
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40
                        position = tuple(map(int, position))

                        cv2.putText(frame, f"{speed:.2f} km/h", position,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.2f} m",
                                    (position[0], position[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_frames.append(frame)
        return output_frames
