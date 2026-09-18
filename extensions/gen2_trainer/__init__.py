"""Gen2 registration. Discovery never loads model weights or optional optimizers."""
from toolkit.extension import Extension


class Gen2TrainerExtension(Extension):
    uid = "gen2_trainer"
    name = "Gen2 style trainer"

    @classmethod
    def get_process(cls):
        from .process import Gen2TrainProcess
        return Gen2TrainProcess


class Gen2V2DiagnosticExtension(Extension):
    uid = "gen2_v2_diagnostic"
    name = "Gen2 v2 activator diagnostic"

    @classmethod
    def get_process(cls):
        from .v2.diagnostic_process import V2DiagnosticProcess
        return V2DiagnosticProcess


AI_TOOLKIT_EXTENSIONS = [Gen2TrainerExtension, Gen2V2DiagnosticExtension]
