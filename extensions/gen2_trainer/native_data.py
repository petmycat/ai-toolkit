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

    def _remove_file_items(self, items_to_remove):
        # Native latent preparation collects decode failures and then drops
        # them here. Reject before the file list or bucket indices can change;
        # another resolution retaining this path must not hide the failure.
        if items_to_remove:
            paths = ", ".join(str(item.path) for item in items_to_remove)
            raise RuntimeError(
                f"Gen2 abort policy: latent cache preparation failed in dataset {self.dataset_path} "
                f"at resolution {self.dataset_config.resolution} for {paths}; "
                "removing training examples is forbidden"
            )

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(f"Gen2 loader index {index} outside dataset of length {len(self)}")
        return super().__getitem__(index)
