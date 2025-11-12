import cv2
import numpy as np


class ViewTranformer():
    def __init__(self):
        court_width = 68
        court_length = 23.32

        # self.pixel_verticies = np.array([
        #     [110,1035],
        #     [265, 275],
        #     [910, 260],
        #     [1640, 915]
        #     ])
        self.pixel_verticies = np.array([
            [50, 700],
            [200, 200],
            [1200, 200],
            [1350, 700]
        ])

        self.target_verticies = np.array([
            [0,court_width],
            [0,0],
            [court_length,0],
            [court_length, court_width]
        ])

        self.pixel_verticies = self.pixel_verticies.astype(np.float32)
        self.target_verticies = self.target_verticies.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_verticies, self.target_verticies)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_verticies, p, False) >= 0
        print(f"[DEBUG] transform_point(): input={point}, inside={is_inside}")

        if not is_inside:
            print(f"[DEBUG] Point {p} nằm ngoài vùng pixel_verticies => bỏ qua.")
            return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        print(f"[DEBUG] Transform result: {transform_point}")

        return transform_point.reshape(-1, 2)

    # def add_transformed_position_to_track(self, tracks):
    #     for object, object_tracks in tracks.items():
    #         for frame_num, track in enumerate(object_tracks):
    #             for track_id, track_infor in track.items():
    #                 position = track_infor["position_adjusted"]
    #                 position = np.array(position)
    #                 position_transformed = self.transform_point(position)
    #                 if position_transformed is not None:
    #                     position_transformed = position_transformed.squeeze().tolist()
    #                 tracks[object][frame_num][track_id]["position_transformed"] = position_transformed
    def add_transformed_position_to_track(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_infor in track.items():
                    # --- Lấy vị trí gốc ---
                    if "position_adjusted" in track_infor:
                        position = track_infor["position_adjusted"]
                    elif "position" in track_infor:
                        position = track_infor["position"]
                    else:
                        continue

                    # --- Chuyển sang numpy array ---
                    position = np.array(position)

                    # --- Biến đổi toạ độ ---
                    position_transformed = self.transform_point(position)

                    # --- Lưu lại vào track ---
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()

                    tracks[object][frame_num][track_id]["position_transformed"] = position_transformed

                    # --- Debug (tuỳ chọn) ---
                    if object == "ball" and position_transformed is not None:
                        print(f"Ball transformed at frame {frame_num}: {position_transformed}")


