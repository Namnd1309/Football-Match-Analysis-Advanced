import cv2

def read_vd (vd_path):
    cap = cv2.VideoCapture(vd_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_vd (output_vd_frame, vd_output_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(vd_output_path, fourcc, 24, (output_vd_frame[0].shape[1],output_vd_frame[0].shape[0]))
    for frame in output_vd_frame:
        out.write(frame)
    out.release()