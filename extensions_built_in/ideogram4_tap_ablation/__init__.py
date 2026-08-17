from toolkit.extension import Extension


class Ideogram4TapCausalAblationExtension(Extension):
    uid = "ideogram4_tap_causal_ablation"
    name = "Ideogram 4 Tap Causal Ablation"

    @classmethod
    def get_process(cls):
        from .process import Ideogram4TapCausalAblationProcess

        return Ideogram4TapCausalAblationProcess


AI_TOOLKIT_EXTENSIONS = [Ideogram4TapCausalAblationExtension]
