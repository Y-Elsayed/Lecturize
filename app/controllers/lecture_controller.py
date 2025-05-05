import os
class LectureController:
    """
    Orchestrates the lecture enhancement pipeline by coordinating all major components.

    Responsibilities:
    - Runs the instructor masking step using a detection model.
    - Prompts the user to manually select whiteboard corners.
    - Applies homography transformation to straighten the board.
    - Saves the visualization and final processed video.
    """
    def __init__(self, model, masker, transformer, selector, visualizer, writer, polygon_output_path):
        self.model = model
        self.masker = masker
        self.transformer = transformer
        self.selector = selector
        self.visualizer = visualizer
        self.writer = writer
        self.polygon_output_path = polygon_output_path

    def run(self, input_path):
        masked_video = self.writer.cleaned_video_path

        # Skip masking step if the masked video already exists
        if not os.path.exists(masked_video):
            # Ensure directory exists
            os.makedirs(os.path.dirname(masked_video), exist_ok=True)
            masked_video = self.masker.mask_professor(input_path, self.model, masked_video)
        else:
            print(f"Skipped masking: using existing file at {masked_video}")

        # 2. Let user select board points
        first_frame = self.selector.get_first_frame(masked_video)
        src_pts = self.selector.select_points(first_frame)

        # 3. Transform video to smartboard format
        self.transformer.compute_homography(src_pts)
        warped_frames = self.transformer.warp_video(masked_video)

        # 4. Save visualizations and output
        self.visualizer.draw_polygon(first_frame, src_pts, self.polygon_output_path)
        self.writer.write_video(warped_frames)


