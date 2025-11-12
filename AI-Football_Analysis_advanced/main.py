import cv2
import numpy as np
from utils import read_vd, save_vd
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTranformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from Tactical_map import PitchHomography, TacticalMapDrawer
from ultralytics import YOLO
from field_structure_detector import FieldStructureDetector

class BallSpeedEstimator:
    def __init__(self, fps=25):
        self.fps = fps
        self.prev_pos = None
        self.total_distance = 0.0
        self.speed_kmh = 0.0

    def update(self, ball_pos):
        """ball_pos: (x_m, y_m) theo mét"""
        if self.prev_pos is not None:
            dx = ball_pos[0] - self.prev_pos[0]
            dy = ball_pos[1] - self.prev_pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            self.total_distance += dist

            speed_m_s = dist * self.fps  # m/s
            self.speed_kmh = speed_m_s * 3.6  # km/h

        self.prev_pos = ball_pos
        return self.speed_kmh, self.total_distance

def main():
    last_ball_seen_frame = -1
    goal_display_counter = 0
    goal_team = None

    # === Load video & models ===
    video_frames = read_vd(
        r"C:\Users\TGDD\Downloads\AI-Football_Analysis_advanced\Data_set\demo_vid_1.mp4"
    )
    fps = 25
    model_keypoints = YOLO(
        r"C:\Users\TGDD\Downloads\AI-Football_Analysis_advanced\Data_set\Yolo8M Field Keypoints\weights\best.pt"
    )

    tracker = Tracker(
        r"C:\Users\TGDD\Downloads\AI-Football_Analysis_advanced\Data_set\Yolo8L Players\weights\best.pt"
    )
    homography = PitchHomography(
        json_path=r"C:\Users\TGDD\Downloads\AI-Football_Analysis_advanced\Data_set\pitch map labels position.json",
        yaml_path=r"C:\Users\TGDD\Downloads\AI-Football_Analysis_advanced\Data_set\config pitch dataset.yaml"
    )
    drawer = TacticalMapDrawer(
        r"C:\Users\TGDD\Downloads\AI-Football_Analysis_advanced\Data_set\tactical map.jpg"
    )
    tracks = tracker.get_object_track(
        video_frames,
        read_from_stub=True,
        # stub_path=r"C:\Users\TGDD\Downloads\AI-Football_Analysis_advanced\stubs\track_stubs_video_test_2.pkl",
        stub_path=r"C:\Users\TGDD\Downloads\AI-Football_Analysis_advanced\stubs\track_stubs_video_test_5.pkl",
    )
    tracker.add_position_to_tracks(tracks)
    tracker.add_transformed_positions(tracks, homography)
    print("[DEBUG] Sample ball:", tracks["ball"][0].get(1, {}))

    # === Camera movement compensation ===
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        # stub_path=r"C:\Users\TGDD\Downloads\AI-Football_Analysis_advanced\stubs\camera_movement_test_stub_2.pkl",
        stub_path=r"C:\Users\TGDD\Downloads\AI-Football_Analysis_advanced\stubs\camera_movement_test_stub_5.pkl"
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement_per_frame
    )

    # === Ball interpolation & transforms ===
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    view_transformer = ViewTranformer()
    view_transformer.add_transformed_position_to_track(tracks)

    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_track(tracks)

    # === Team assignment ===
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    # === Ball control ===
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_dict = tracks["ball"][frame_num]
        if 1 in ball_dict:
            ball_bbox = ball_dict[1]["bbox"]
        else:
            ball_bbox = None

        if ball_bbox is not None:
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        else:
            assigned_player = -1

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(1)

    team_ball_control = np.array(team_ball_control)

    # === Annotate video frames ===
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    field_detector = FieldStructureDetector(homography.keypoints_dict)

    team1_ratio_list, team2_ratio_list = [], []
    for f in range(len(team_ball_control)):
        team1_possession = np.sum(team_ball_control[: f + 1] == 1)
        team2_possession = np.sum(team_ball_control[: f + 1] == 2)
        total = f + 1
        team1_ratio_list.append((team1_possession / total) * 100)
        team2_ratio_list.append((team2_possession / total) * 100)

    # === Prepare for drawing ===
    combined_frames = []
    ball_speed_estimator = BallSpeedEstimator(fps=fps)
    ball_path_points = []  # lưu quỹ đạo bóng

    # === Main loop ===
    for i, frame in enumerate(output_video_frames):
        results_keypoints = model_keypoints(frame, conf=0.7)
        bboxes_k_c = results_keypoints[0].boxes.xywh.cpu().numpy()
        labels_k = list(results_keypoints[0].boxes.cls.cpu().numpy())

        player_tracks = tracks["players"][i]
        detected_ppos_src_pts, player_colors = [], []
        for player_id, player_data in player_tracks.items():
            if "position_adjusted" in player_data:
                detected_ppos_src_pts.append(player_data["position_adjusted"])
                player_colors.append(player_data["team_color"])
        detected_ppos_src_pts = np.array(detected_ppos_src_pts)

        # === Xác định vị trí cầu thủ trên tactical map ===
        if len(labels_k) > 3 and len(detected_ppos_src_pts) > 0:
            h = homography.compute_homography(bboxes_k_c, labels_k)
            pred_dst_pts = homography.transform_points(detected_ppos_src_pts)
        else:
            pred_dst_pts = []
            player_colors = []

        # === Lấy vị trí bóng và chuyển sang pixel trên tactical map ===
        ball_point = None
        speed_kmh, dist_m = 0.0, 0.0

        print(f"[DEBUG] Frame {i}: ball in tracks? {'ball' in tracks}")
        if "ball" in tracks:
            print(
                f"[DEBUG] tracks['ball'][{i}] keys: {list(tracks['ball'][i].keys()) if i < len(tracks['ball']) else 'out of range'}")
        if "ball" in tracks and i < len(tracks["ball"]):
            ball_info = tracks["ball"][i].get(1, {})
            if "position_transformed" in ball_info and ball_info["position_transformed"] is not None:
                bx_m, by_m = map(float, ball_info["position_transformed"])

                # Tính tốc độ & quãng đường
                speed_kmh, dist_m = ball_speed_estimator.update((bx_m, by_m))

                # Quy đổi sang pixel tactical map
                court_length, court_width = 23.32, 68
                map_h, map_w = drawer.map_img.shape[:2]
                x_px = int((bx_m / court_length) * map_w)
                y_px = int(map_h - (by_m / court_width) * map_h)
                ball_point = (x_px, y_px)

        # === Vẽ tactical map (1 lần duy nhất mỗi frame) ===
        print(f"[DEBUG] Frame {i}: ball_point = {ball_point}")
        tac_map_frame = drawer.draw_positions(pred_dst_pts, player_colors, ball_point)

        # === Hiển thị tốc độ & quãng đường bóng (nếu có) ===
        if ball_point is not None:
            cv2.putText(
                tac_map_frame, f"Speed: {speed_kmh:.1f} km/h",
                (ball_point[0] + 10, ball_point[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            cv2.putText(
                tac_map_frame, f"Distance: {dist_m:.2f} m",
                (ball_point[0] + 10, ball_point[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2
            )

        # === Combine tactical map + video ===
        tac_map_frame = cv2.resize(tac_map_frame, (frame.shape[1] // 3, frame.shape[0]))
        frame_resized = cv2.resize(frame, (frame.shape[1] * 2 // 3, frame.shape[0]))
        combined = np.concatenate((frame_resized, tac_map_frame), axis=1)
        combined_frames.append(combined)

    # === Save video output ===
    save_vd(
        combined_frames,
        r"C:/Users/TGDD/Downloads/AI-Football_Analysis_advanced/output_video/output_with_ball_speed_v2.avi",
    )


if __name__ == "__main__":
    main()
