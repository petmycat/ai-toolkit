"""Ideogram 4 Gen2 style trainer extension."""

from toolkit.extension import Extension


class Gen2TrainerExtension(Extension):
    uid = "gen2_trainer"
    name = "Ideogram 4 Gen2 Trainer"

    @classmethod
    def get_process(cls):
        from .Gen2Trainer import Gen2Trainer

        return Gen2Trainer


AI_TOOLKIT_EXTENSIONS = [Gen2TrainerExtension]
