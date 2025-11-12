import cv2, yaml, json, numpy as np

class PitchHomography:
    def __init__(self, json_path, yaml_path):
        with open(json_path, 'r') as f:
            self.map_points = json.load(f)
        with open(yaml_path, 'r') as f:
            self.yaml_data = yaml.safe_load(f)['names']

        # Sắp xếp các điểm keypoint đúng thứ tự
        self.map_points_sorted = {self.yaml_data[i]: self.map_points[self.yaml_data[i]] for i in self.yaml_data}

        # Gán để class khác (như FieldStructureDetector) có thể truy cập
        self.keypoints_dict = self.map_points_sorted

        self.h_previous = None  # giữ ma trận homography cũ
        self.update_threshold = 100  # ngưỡng để cập nhật homography mới

    def compute_homography(self, keypoints, labels):
        if len(labels) < 4:
            return self.h_previous

        detected_labels = [self.yaml_data[i] for i in labels]
        src_pts = np.array([keypoints[i][:2] for i in range(len(keypoints))])
        dst_pts = np.array([self.map_points_sorted[label] for label in detected_labels])

        h, mask = cv2.findHomography(src_pts, dst_pts)
        if self.h_previous is not None and h is not None:
            diff = np.linalg.norm(self.h_previous - h)
            if diff < self.update_threshold:
                return self.h_previous  # giữ lại ma trận cũ nếu thay đổi nhỏ
        self.h_previous = h
        return h

    def transform_points(self, points):
        if self.h_previous is None or len(points) == 0:
            return []
        transformed = []
        for pt in points:
            p = np.array([pt[0], pt[1], 1.0])
            p_t = self.h_previous @ p
            p_t /= p_t[2]
            transformed.append((int(p_t[0]), int(p_t[1])))
        return transformed
