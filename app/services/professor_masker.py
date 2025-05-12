import cv2
import numpy as np
from tqdm import tqdm

class ProfessorMasker:
    def __init__(self, conf_threshold=0.5, iou_threshold=0.4, target_class=0, padding = 80):
        """
        Initializes the ProfessorMasker with detection parameters.
        - conf_threshold: Confidence threshold for detection.
        - iou_threshold: IoU threshold for non-max suppression.
        - target_class: Class index to be detected (default is 0 = person in COCO).
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_class = target_class
        self.padding = padding
        self.board_canvas = None      # Holds the clean board as it is built
        self.freeze_mask = None       # Tracks areas to "freeze" (professor coverage)
        self.initialized = False

    def _is_solid_color_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.all(gray == gray[0, 0])

    def _create_freeze_mask(self, boxes, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, int(x1) - self.padding)
            y1 = max(0, int(y1) - self.padding)
            x2 = min(int(x2) + self.padding, width)
            y2 = min(int(y2) + self.padding, height)
            mask[y1:y2, x1:x2] = 1
        return mask
    
    def _blend_region(self, base, overlay, mask, feather=10):
        """
        Blend overlay into base using a feathered mask.
        - base: the target frame (board_canvas)
        - overlay: the current raw frame
        - mask: binary mask of the freeze region
        - feather: number of pixels to feather at the edges
        """
        kernel = np.ones((feather, feather), np.uint8)
        soft_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        soft_mask = cv2.GaussianBlur(soft_mask.astype(np.float32), (feather * 2 + 1, feather * 2 + 1), 0)

        max_val = soft_mask.max()
        if max_val == 0:
            return base.copy()  # Nothing to blend — return original frame

        soft_mask = np.clip(soft_mask / max_val, 0, 1)  # Normalize safely

        blended = base.copy()
        for c in range(3):
            blended[..., c] = soft_mask * base[..., c] + (1 - soft_mask) * overlay[..., c]

        return blended.astype(np.uint8)




    def mask_professor(self, input_path, model, output_path, max_buffer_size=100):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video at: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Run YOLOv8 tracking
        results = model.track_video(
            input_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.target_class]
        )

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for i, result in enumerate(tqdm(results, total=frame_count, desc="Masking professor")):
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []
            raw_frame = result.orig_img.copy()

            if not self.initialized:
                if self._is_solid_color_frame(raw_frame):
                    continue  # Skip solid black/empty frames
                self.board_canvas = raw_frame.copy()
                self.freeze_mask = np.zeros((height, width), dtype=np.uint8)
                self.initialized = True

            # Create current frame's freeze mask (where the professor is)
            current_freeze_mask = self._create_freeze_mask(boxes, height, width)

            # Compute mask of pixels that are not frozen
            updatable_mask = (self.freeze_mask == 0) & (current_freeze_mask == 0)

            # Create a binary mask for updatable regions (1 = frozen area)
            freeze_mask_inv = np.logical_not(updatable_mask).astype(np.uint8)

            # Blend only where the patch borders meet live frame
            self.board_canvas = self._blend_region(self.board_canvas, raw_frame, freeze_mask_inv, feather=10)


            # Write current board canvas as output
            output_frame = self.board_canvas.copy()
            out.write(output_frame)

            # Freeze current regions with professor for next frame
            self.freeze_mask = current_freeze_mask.copy()

        out.release()
        print(f"\nMasked video saved at: {output_path}")
        return output_path
