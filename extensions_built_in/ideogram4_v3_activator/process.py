from collections import OrderedDict

from jobs.process import BaseExtensionProcess

from .config import load_config
from .pipeline import Ideogram4V3ActivatorPipeline


class Ideogram4V3ActivatorPipelineProcess(Ideogram4V3ActivatorPipeline, BaseExtensionProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.v3_config = load_config(self.get_conf("ideogram4_v3_activator", {}))
        if self.v3_config.enabled:
            self.initialize_pipeline()

    def run(self):
        super().run()
        if self.v3_config.enabled:
            self.run_pipeline()
