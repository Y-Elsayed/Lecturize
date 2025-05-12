from app.models.professor_detection_model import ProfessorDetectionModel
from app.services.professor_masker import ProfessorMasker
from app.services.whiteboard_transformer import WhiteboardTransformer
from app.views.frame_selector import FrameSelector
from app.views.visualizer import Visualizer
from app.views.video_writer import VideoWriter
from app.controllers.lecture_controller import LectureController
from app.utils.config_loader import load_config
from ui.config_updater import run_ui_and_update_config


def main():
    # Step 1: Launch UI to update config
    run_ui_and_update_config(
        input_path='input_videos/lecture.mp4',
        output_dir='output'
    )


    # Step 2: Load the updated configuration
    config = load_config('configs/config.yaml')

    input_video_path = config['video']['input_path']
    cleaned_video_path = config['output']['cleaned_board_video']
    smart_board_model_path = config['output']['smartboard_video']

    # Step 3: Initialize core components
    model = ProfessorDetectionModel(config['model']['name'])
    masker = ProfessorMasker(
        conf_threshold=config['model']['confidence_threshold'],
        iou_threshold=config['model']['iou_threshold'],
        target_class=config['model']['target_class']
    )
    transformer = WhiteboardTransformer()
    selector = FrameSelector()
    visualizer = Visualizer()
    writer = VideoWriter(cleaned_video_path,smart_board_model_path)
    writer.set_fps_from_video(input_video_path)

    # Step 4: Run the lecture enhancement pipeline
    controller = LectureController(
        model=model,
        masker=masker,
        transformer=transformer,
        selector=selector,
        visualizer=visualizer,
        writer=writer,
        polygon_output_path=config['output']['polygon_image'] 
    )

    controller.run(input_video_path)


if __name__ == '__main__':
    main()
