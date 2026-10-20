"""Fused-PRoPE camera-control entry point for ReCam training.

The camera variant is a config schema, not a trainer subclass:
``ReCamTrainer`` reads an optional ``camera`` section off whatever config
class it is handed and threads it through the dataset, the model and the
checkpoint. Only the CLI schema differs, so that is all this module selects.
"""

from example.recam.config.camera_config import CameraExpConfig
from example.recam.trainers.trainer import main


if __name__ == "__main__":
    main(CameraExpConfig)
