import cv2
import numpy as np

class FieldStructureDetector:
    def __init__(self, keypoints_json):
        self.keypoints = keypoints_json
        self.define_field_zones()

    def define_field_zones(self):
        k = self.keypoints
        self.goal_left = np.array([k["TL6ML"], k["TL6MC"], k["BL6MC"], k["BL6ML"]])
        self.goal_right = np.array([k["TR6ML"], k["TR6MC"], k["BR6MC"], k["BR6ML"]])
        self.boundary = np.array([k["TLC"], k["TRC"], k["BRC"], k["BLC"]])

    def check_goal(self, ball_position):
        if cv2.pointPolygonTest(self.goal_right, ball_position, False) >= 0:
            return "goal_right"
        elif cv2.pointPolygonTest(self.goal_left, ball_position, False) >= 0:
            return "goal_left"
        return None

    def check_out_of_play(self, ball_position):
        return cv2.pointPolygonTest(self.boundary, ball_position, False) < 0
