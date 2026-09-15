"""Importable by spawned workers; native preprocessing is inherited unchanged."""
from toolkit.data_loader import AiToolkitDataset


class Gen2AbortDataset(AiToolkitDataset):
    def __init__(self, dataset_config, batch_size=1, sd=None):
        from .diagnostics import isolated_rng
        # Cold-cache preparation must not perturb the next dataset's bucket
        # initialization or subsequent training noise draws.
        with isolated_rng(dataset_config.gen2_initialization_seed):
            super().__init__(dataset_config, batch_size=batch_size, sd=sd)

    def _get_replacement_index(self, index):
        raise RuntimeError(
            f"Gen2 abort policy: native image load failed for {self.file_list[index].path}; replacement is forbidden"
        )

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(f"Gen2 loader index {index} outside dataset of length {len(self)}")
        return super().__getitem__(index)
