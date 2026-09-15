"""Gen2 registration. Discovery never loads model weights or optional optimizers."""
from toolkit.extension import Extension


class Gen2TrainerExtension(Extension):
    uid = "gen2_trainer"
    name = "Gen2 style trainer"

    @classmethod
    def get_process(cls):
        from .process import Gen2TrainProcess
        return Gen2TrainProcess


AI_TOOLKIT_EXTENSIONS = [Gen2TrainerExtension]
