import cv2
import numpy as np

class Visualizer:
    """
    Visualizer
    ----------
    Handles drawing visual elements like point markers and polygons on video frames
    and saving them as annotated images (useful for presentations or debugging).
    """
    def draw_polygon(self, frame, points, output_path):
        copy = frame.copy()
        points_array = np.array(points, dtype=np.int32)
        for pt in points:
            cv2.circle(copy, tuple(pt), 5, (0, 255, 0), -1)
        cv2.polylines(copy, [points_array], True, (0, 0, 255), 2)
        cv2.imwrite(output_path, copy)