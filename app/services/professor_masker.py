import cv2
import numpy as np
import os
from tqdm import tqdm

class ProfessorMasker:
    def __init__(self, conf_threshold=0.5, iou_threshold=0.4, target_class=0):
        """
        Initializes the ProfessorMasker with detection parameters.

        Parameters:
        - conf_threshold: Confidence threshold for detection.
        - iou_threshold: IoU threshold for non-max suppression.
        - target_class: Class index to be detected (default is 0 = person in COCO).
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_class = target_class

    def mask_professor(self, input_path, model, output_path):
        """
        Removes the professor from the lecture video frames using detection bounding boxes.

        Responsibilities:
        - Uses the tracking model to detect the professor in each frame.
        - Applies masking logic to overwrite detected regions with background content.
        - Saves the cleaned video with minimal occlusion over the board.
        """
        # Dynamically get input video properties
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open input video at: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        results = model.track_video(
            input_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.target_class]
        )

        video_writer = None
        old_frame = None

        print(f"Processing and masking video: {input_path}")
        for i, result in enumerate(tqdm(results, total=frame_count, desc="Masking professor")):
            raw_frame = result.orig_img
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []

            if video_writer is None:
                h, w = raw_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                old_frame = raw_frame.copy()

            # Start with a full update from the new frame
            updated = raw_frame.copy()

            # For each detected professor box, preserve the old frame content
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                updated[y1:y2, x1:x2] = old_frame[y1:y2, x1:x2]


            old_frame = updated.copy()
            video_writer.write(updated)

        video_writer.release()
        print(f"Masked video saved to: {output_path}")
        return output_path
