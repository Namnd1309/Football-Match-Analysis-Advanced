import cv2
import numpy as np

class TacticalMapDrawer:
    def __init__(self, tactical_map_path=None, map_size=(680, 1050)):
        if tactical_map_path:
            self.map_img = cv2.imread(tactical_map_path)
        else:
            self.map_img = self._create_blank_pitch(map_size)
        self.ball_traj = []  # lưu các điểm quỹ đạo bóng (pixel trên map)

    def _create_blank_pitch(self, size):
        w, h = size
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        line_color = (0, 0, 0)
        thickness = 2
        cv2.rectangle(img, (50, 50), (w - 50, h - 50), line_color, thickness)
        cv2.line(img, (w // 2, 50), (w // 2, h - 50), line_color, thickness)
        cv2.circle(img, (w // 2, h // 2), 70, line_color, thickness)
        cv2.rectangle(img, (w // 2 - 100, 50), (w // 2 + 100, 150), line_color, thickness)
        cv2.rectangle(img, (w // 2 - 100, h - 150), (w // 2 + 100, h - 50), line_color, thickness)
        return img

    def draw_positions(self, player_points, player_colors=None, ball_point=None):
        img = self.map_img.copy()

        # --- Vẽ cầu thủ ---
        if player_points is not None:
            for i, pt in enumerate(player_points):
                color = (0, 0, 255)
                if player_colors is not None and i < len(player_colors):
                    color = tuple(int(c) for c in player_colors[i])
                cv2.circle(img, (int(pt[0]), int(pt[1])), 6, color, -1)

        # --- Vẽ bóng + quỹ đạo ---
        if ball_point is not None:
            bx, by = int(ball_point[0]), int(ball_point[1])
            self.ball_traj.append((bx, by))
            if len(self.ball_traj) > 100:
                self.ball_traj.pop(0)

        if len(self.ball_traj) > 1:
            for j in range(1, len(self.ball_traj)):
                cv2.line(img, self.ball_traj[j - 1], self.ball_traj[j], (0, 0, 200), 2)

        if ball_point is not None:
            cv2.circle(img, (int(ball_point[0]), int(ball_point[1])), 8, (0, 0, 255), -1)

        return img
