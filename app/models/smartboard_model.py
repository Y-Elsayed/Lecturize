from ultralytics import YOLO
import torch

class SmartBoardModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def track_video(self, input_path, conf=0.3, iou=0.5, classes=None):
        """
        Runs object tracking on the given video using the detection model.

        Parameters:
        - input_path: Path to the input video.
        - conf: Confidence threshold for detection.
        - iou: IoU threshold.
        - classes: List of class indices to detect.
        """
        return self.model.track(
            source=input_path,
            show=False,
            conf=conf,
            iou=iou,
            classes=classes,
            stream=True
        )
