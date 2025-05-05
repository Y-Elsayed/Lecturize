import cv2
import numpy as np

class FrameSelector:
    """
    Allows the user to interactively select four points from a video frame using mouse clicks.

    This is typically used to identify the corners of a whiteboard for homography transformation.
    """
    def __init__(self):
        self.selected_points = []
        self.window_name = 'Select 4 points (Top-Left, Top-Right, Bottom-Right, Bottom-Left)'

    def _click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_points.append((x, y))
            cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(self.window_name, param)

    def select_points(self, frame):
        """
        Opens a window to display the given frame and lets the user click to select four points.

        Args:
            frame (ndarray): The image frame from which to select points.

        Returns:
            list: A list of four (x, y) tuples representing selected points.
        """
        self.selected_points.clear()
        clone = frame.copy()
        cv2.imshow(self.window_name, clone)
        cv2.setMouseCallback(self.window_name, self._click_event, clone)

        print("Click on the 4 corners of the whiteboard (Top-Left, Top-Right, Bottom-Right, Bottom-Left)")
        while True:
            if len(self.selected_points) == 4:
                break
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

        cv2.destroyAllWindows()
        return self.selected_points
    def get_first_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Could not read the first frame from video.")
        return frame

