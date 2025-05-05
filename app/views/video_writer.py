import cv2

class VideoWriter:
    """
    Handles saving processed frames as video files.
    Supports dynamic FPS extraction from a reference video if not provided.
    """
    def __init__(self, cleaned_video_path, smartboard_path, fps=None):
        self.cleaned_video_path = cleaned_video_path
        self.smartboard_path = smartboard_path
        self.fps = fps

    def set_fps_from_video(self, reference_video_path):
        """
        Sets the FPS value by reading it from the given video.
        """
        cap = cv2.VideoCapture(reference_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video at {reference_video_path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    def write_video(self, frames):
        if not frames:
            raise ValueError("No frames to write")
        if self.fps is None:
            raise ValueError("FPS must be set manually or via set_fps_from_video()")

        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(self.smartboard_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"Video saved at: {self.smartboard_path}")
