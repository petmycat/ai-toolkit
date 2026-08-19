from toolkit.extension import Extension


class Ideogram4V3RouterPhaseCExtension(Extension):
    uid = "ideogram4_v3_router_phase_c"
    name = "Ideogram 4 V3 Router Phase C"

    @classmethod
    def get_process(cls):
        from .process import Ideogram4V3RouterPhaseCProcess

        return Ideogram4V3RouterPhaseCProcess


AI_TOOLKIT_EXTENSIONS = [Ideogram4V3RouterPhaseCExtension]
