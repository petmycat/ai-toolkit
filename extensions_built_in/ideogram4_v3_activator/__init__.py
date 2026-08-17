from toolkit.extension import Extension


class Ideogram4V3ActivatorPipelineExtension(Extension):
    uid = "ideogram4_v3_activator"
    name = "Ideogram 4 V3 Activator Pipeline"

    @classmethod
    def get_process(cls):
        from .process import Ideogram4V3ActivatorPipelineProcess

        return Ideogram4V3ActivatorPipelineProcess


AI_TOOLKIT_EXTENSIONS = [Ideogram4V3ActivatorPipelineExtension]
