import cv2
import numpy as np

class WhiteboardTransformer:
    """
    Handles homography transformation to convert a tilted whiteboard view into a frontal smartboard-like view.

    Responsibilities:
    - Computes output dimensions based on selected board corners.
    - Calculates homography matrix from selected points.
    - Applies perspective warp to each video frame.
    """
    def __init__(self):
        self.M = None
        self.output_size = None

    def compute_output_size(self, pts):
        tl, tr, br, bl = [np.array(p, dtype=np.float32) for p in pts]
        w1 = np.linalg.norm(tr - tl)
        w2 = np.linalg.norm(br - bl)
        h1 = np.linalg.norm(bl - tl)
        h2 = np.linalg.norm(br - tr)
        return int(max(w1, w2)), int(max(h1, h2))


    def compute_homography(self, src_pts):
        src_pts = np.array(src_pts, dtype=np.float32).reshape((4, 2))
        self.output_size = self.compute_output_size(src_pts)
        dst_pts = np.array([
            [0, 0],
            [self.output_size[0] - 1, 0],
            [self.output_size[0] - 1, self.output_size[1] - 1],
            [0, self.output_size[1] - 1]
        ], dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)


    def warp_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            warped = cv2.warpPerspective(frame, self.M, self.output_size)
            frames.append(warped)
        cap.release()
        return frames