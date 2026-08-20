from toolkit.extension import Extension


class Ideogram4V3JointGateOracleC0Extension(Extension):
    uid = "ideogram4_v3_joint_gate_oracle_c0"
    name = "Ideogram 4 V3 Joint Gate Oracle C0"

    @classmethod
    def get_process(cls):
        from .process import Ideogram4V3JointGateOracleC0Process

        return Ideogram4V3JointGateOracleC0Process


AI_TOOLKIT_EXTENSIONS = [Ideogram4V3JointGateOracleC0Extension]
