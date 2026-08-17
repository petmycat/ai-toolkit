from toolkit.extension import Extension


class Ideogram4V3ResidualAblationExtension(Extension):
    uid = "ideogram4_v3_residual_ablation"
    name = "Ideogram 4 V3 Residual Ablation"

    @classmethod
    def get_process(cls):
        from .process import Ideogram4V3ResidualAblationProcess

        return Ideogram4V3ResidualAblationProcess


AI_TOOLKIT_EXTENSIONS = [Ideogram4V3ResidualAblationExtension]
