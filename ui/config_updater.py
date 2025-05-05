import argparse
import yaml

CONFIG_PATH = 'configs/config.yaml'

def update_config(input_path=None, output_dir=None, fps=None):
    """
    Config Updater
    --------------
    Command-line interface for updating the YAML config file with new input paths,
    output directories, or FPS settings before launching the pipeline.
    """
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file) or {}

    if 'video' not in config:
        config['video'] = {}

    if input_path:
        config['video']['input_path'] = input_path
    if output_dir:
        config['video']['output_dir'] = output_dir

    with open(CONFIG_PATH, 'w') as file:
        yaml.dump(config, file)

    print("Config file updated successfully.")


def run_ui_and_update_config(input_path=None, output_dir=None, fps=None):
    # commented the CLI part for now, as the UI is not implemented yet

    # parser = argparse.ArgumentParser(description="Update config file before running the pipeline")
    # parser.add_argument('--input', help='Path to input video')
    # parser.add_argument('--output', help='Output directory')
    # parser.add_argument('--fps', type=int, help='Frames per second')

    # args = parser.parse_args()
    # update_config(input_path=args.input, output_dir=args.output, fps=args.fps)
    update_config(input_path=input_path, output_dir=output_dir, fps=fps)
