import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import pickle
import cv2
import os
import pandas as pd
import sys
sys.path.append("../")
from utils import get_bbox_width,get_center_of_bbox, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]["position"] = position

    # def interpolate_ball_positions(self, ball_positions):
    #     ball_positions = [x.get(1,{}).get("bbox", []) for x in ball_positions]
    #     df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
    #
    #     #interpolated missing value
    #     df_ball_positions = df_ball_positions.interpolate()
    #     df_ball_positions = df_ball_positions.bfill()
    #
    #     ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
    #
    #     return ball_positions
    def interpolate_ball_positions(self, ball_positions_raw):
        ball_positions = []
        for frame in ball_positions_raw:  # đổi tên biến để dễ hiểu
            if len(frame) > 0:
                first_key = list(frame.keys())[0]
                ball_positions.append(frame[first_key].get("bbox", []))
            else:
                ball_positions.append([])

        # Nếu không có bbox hợp lệ thì bỏ qua tránh lỗi
        if not any(len(b) == 4 for b in ball_positions):
            print("⚠️ No valid ball positions found — skipping interpolation")
            return [{1: {"bbox": [0, 0, 0, 0]}} for _ in ball_positions]

        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate().bfill()
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for i in range(0,len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_track(self, frames, read_from_stub = False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frames_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            print(cls_names)

            #convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #conver goal keeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            #Track_obj
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frames_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frames_num][track_id] = {"bbox": bbox}

            # Track bóng bằng ByteTrack riêng (nếu YOLO detect được)
            ball_detections = [
                d for d in detection_supervision if d[3] == cls_names_inv["ball"]
            ]

            if len(ball_detections) > 0:
                ball_detection_supervision = sv.Detections(
                    xyxy=np.array([b[0] for b in ball_detections]),
                    confidence=np.ones(len(ball_detections)),
                    class_id=np.full(len(ball_detections), cls_names_inv["ball"]),
                )
                ball_with_tracks = self.tracker.update_with_detections(ball_detection_supervision)
                for ball_det in ball_with_tracks:
                    bbox = ball_det[0].tolist()
                    track_id = ball_det[4]
                    tracks["ball"][frames_num][track_id] = {"bbox": bbox}

        if stub_path is not None:
            with open (stub_path, "wb") as f:
                pickle.dump(tracks,f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):

        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(frame,
                    center=(x_center,y2),
                    axes=(int(width), int(0.35*width)),
                    angle=0.0,
                    startAngle= -45,
                    endAngle=235,
                    color = color,
                    thickness= 2,
                    lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width // 2

        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_point = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [triangle_point], 0, color,cv2.FILLED)
        cv2.drawContours(frame, [triangle_point], 0, (0,0,0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        #Draw a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255,255,255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        #Get the number of time each team had the ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(frame, f"Team 1 ball control: {team_1*100:.2f}%", (1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 ball control: {team_2*100:.2f}%", (1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def add_transformed_positions(self, tracks, homography):
        if "ball" not in tracks:
            return  # Không có bóng => bỏ qua

        from utils import get_center_of_bbox
        import numpy as np

        for i, ball_frame in enumerate(tracks["ball"]):
            for track_id, ball_info in ball_frame.items():
                # --- Ưu tiên position_adjusted hoặc position ---
                position = ball_info.get("position_adjusted") or ball_info.get("position")

                # --- Nếu chưa có, lấy tâm bbox ---
                if position is None and "bbox" in ball_info and ball_info["bbox"] is not None:
                    position = get_center_of_bbox(ball_info["bbox"])

                # --- Nếu vẫn không có thì bỏ qua ---
                if position is None:
                    continue

                # --- Đảm bảo là numpy array ---
                position = np.array(position, dtype=np.float32).reshape(1, 2)

                # --- Áp dụng homography để chuyển sang tọa độ sân ---
                transformed_pts = homography.transform_points(position)

                # --- Gán vào tracks để tactical map đọc ---
                if transformed_pts is not None and len(transformed_pts) > 0:
                    tx, ty = transformed_pts[0]
                    tracks["ball"][i][track_id]["position_transformed"] = (float(tx), float(ty))

                    # --- Debug (tuỳ chọn) ---
                    print(f"Ball transformed at frame {i}: ({tx:.2f}, {ty:.2f})")

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            #Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (255,255,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0,0,255))

            #Draw referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (255,0,255))

            # --- Draw ball ---
            # Giữ lịch sử vị trí bóng để vẽ quỹ đạo
            if not hasattr(self, "ball_positions_history"):
                self.ball_positions_history = []

            if ball_dict and len(ball_dict) > 0:
                for track_id, ball in ball_dict.items():
                    if "bbox" in ball:
                        color = (0, 255, 0)
                        frame = self.draw_triangle(frame, ball["bbox"], color)

                        # Lưu tâm bóng để vẽ quỹ đạo
                        cx, cy = get_center_of_bbox(ball["bbox"])
                        self.ball_positions_history.append((cx, cy))

                        # Giữ lại 50 điểm gần nhất cho gọn
                        self.ball_positions_history = self.ball_positions_history[-50:]
            else:
                # không có bóng detect trong frame
                pass

            # --- Vẽ quỹ đạo bay ---
            if hasattr(self, "ball_positions_history"):
                for j in range(1, len(self.ball_positions_history)):
                    cv2.line(frame,
                             self.ball_positions_history[j - 1],
                             self.ball_positions_history[j],
                             (0, 255, 0), 2)

            #Draw t eam ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
