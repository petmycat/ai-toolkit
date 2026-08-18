from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from dataclasses import asdict
from typing import Any, Dict

import yaml

from toolkit.paths import TOOLKIT_ROOT

from .config import StageConfig


class Ideogram4V3ActivatorStageProcess:
    def __init__(self, pipeline, stage: StageConfig):
        self.pipeline = pipeline
        self.stage = stage

    @property
    def stage_root(self) -> str:
        return os.path.join(self.pipeline.run_root, self.stage.name)

    @property
    def snapshot_path(self) -> str:
        return os.path.join(self.pipeline.snapshot_root, f"{self.stage.name}.yaml")

    @property
    def contract_path(self) -> str:
        return os.path.join(self.pipeline.contract_root, f"{self.stage.name}.json")

    def _caption_sources(self) -> Dict[str, Any]:
        source = copy.deepcopy(self.pipeline.v3_config.dataset_schedule)
        source.setdefault("enabled", True)
        return source

    def _internal_three_phase(self) -> Dict[str, Any]:
        parent = copy.deepcopy(self.pipeline.raw_process_config.get("three_phase_trigger_training", {}))
        trigger = copy.deepcopy(parent.get("trigger", {}))
        trigger.setdefault("placeholder", "[trigger]")
        trigger.setdefault("literal", self.pipeline.literal)
        trigger.setdefault("span_detection", "offsets")
        trigger.setdefault("mask_all_occurrences", True)
        trigger.setdefault("occurrence_mode", "additive")
        activator = copy.deepcopy(parent.get("text_activator", {}))
        activator.setdefault("architecture_mode", "module_lora")
        activator.setdefault("embedding", {"enabled": True, "tokens": 4, "init_mode": "literal_initialization", "init_words": "", "dtype": "bf16"})
        activator.setdefault("te_adapter", {"enabled": True, "mode": "module_lora", "rank": 4, "alpha": 4, "dropout": 0.0, "target_modules": ["down_proj"]})
        activator["embedding"].update({"enabled": True, "tokens": 4, "init_mode": "literal_initialization"})
        activator["te_adapter"]["enabled"] = True
        activator["tap_adapters"] = {"enabled": False}

        is_a1 = self.stage.name == "semantic_activator"
        a1_source = self.pipeline.a1_artifact_paths()
        phase = {
            "enabled": True,
            "steps": self.stage.steps,
            "optimizer": self.stage.optimizer,
            "optimizer_params": copy.deepcopy(self.stage.optimizer_params),
            "learning_rates": copy.deepcopy(self.stage.learning_rates),
            "train": {
                "embedding": is_a1,
                "te_adapter": True,
                "tap_adapters": False,
                "diffusion_lora": False,
            },
            "caption_sources": self._caption_sources(),
                "losses": {
                    (
                        "semantic_helper_consistency"
                        if is_a1 else "direct_v3_reconstruction"
                    ): {"enabled": True, "weight": 1.0},
                },
            "save_steps": list(self.stage.save_steps),
        }
        if not is_a1:
            phase["text_activator_source"] = {"path": a1_source["embedding"], "step": "final"}
            phase["diffusion_lora_source"] = {"path": self.pipeline.v3_weights_path, "step": "final"}

        artifacts = {
            "output_root": self.pipeline.run_root,
            "phase_a1": asdict(self.stage.artifacts),
            "phase_a2": asdict(self.stage.artifacts),
        }
        runtime_sources = {
            "embedding": None if is_a1 else a1_source["embedding"],
            "te_adapter": None if is_a1 else a1_source["te_adapter"],
            "tap_adapters": None,
            "diffusion_lora": None if is_a1 else self.pipeline.v3_weights_path,
        }
        internal_phase = "a1" if is_a1 else "a2"
        block = {
            "enabled": True,
            "schema_version": 8,
            "objective_mode": "ideogram4_v3_activator",
            "execution": {"start_phase": internal_phase, "stop_after_phase": internal_phase},
            "trigger": trigger,
            "text_activator": activator,
            "reachability_probe": {"enabled": True},
            "data_split": {
                "enabled": True,
                "manifest_path": self.pipeline.v3_config.fixed_validation["data_split_manifest"],
                "seed": int(self.pipeline.v3_config.fixed_validation["seed"]),
            },
            "validation": {
                "enabled": True,
                "steps": list(self.stage.validation_steps),
                "comparison_variants": [
                    "A_base_activator_no_v3",
                    "B_v3_original_literal",
                    "C_v3_current_activator",
                    "D_v3_each_helper_no_grad",
                ],
                "seed": int(self.pipeline.v3_config.fixed_validation["seed"]),
                "fixed_timesteps": list(self.pipeline.v3_config.fixed_validation["fixed_timesteps"]),
                "data_split_manifest": self.pipeline.v3_config.fixed_validation["data_split_manifest"],
                "caption_sources": ["structured"],
                "negative_phrases": list(
                    self.pipeline.v3_config.fixed_validation.get(
                        "negative_phrases",
                        ["photorealistic photograph", "technical drawing"],
                    )
                ),
                "heldout_output_filename": self.stage.artifacts.validation_file,
            },
            "phase_a1": phase if is_a1 else {"enabled": False},
            "phase_b": {"enabled": False},
            "phase_a2": phase if not is_a1 else {"enabled": False},
            "artifacts": artifacts,
            "runtime": {
                "active_phase": internal_phase,
                "orchestrated": True,
                "run_root": self.pipeline.run_root,
                "config_snapshot": self.snapshot_path,
                "completion_contract": self.contract_path,
            },
            "phase_runtime": {
                "caption_sources": self._caption_sources(),
                "losses": copy.deepcopy(phase["losses"]),
                "save_steps": list(self.stage.save_steps),
                "sources": runtime_sources,
                "objective": {
                    "name": self.stage.name,
                    "pipeline": "ideogram4_v3_activator",
                    "structured_json_only": is_a1,
                    "helper_schedule": copy.deepcopy(self.pipeline.v3_config.helper_schedule),
                    "dataset_schedule": copy.deepcopy(self.pipeline.v3_config.dataset_schedule),
                    "embedding_frozen": not is_a1,
                    "te_only": not is_a1,
                    "v3_active_frozen": not is_a1,
                    "fresh_optimizer": True,
                    "best_metric": "heldout_loss" if not is_a1 else None,
                    "literal_initialization": copy.deepcopy(self.pipeline.v3_config.literal_initialization),
                    "literal_initialization_root": os.path.join(self.pipeline.run_root, "phase_a1"),
                    "prototype_ema_decay": 0.95,
                    "prototype_weight": 0.1,
                    "disturbance_max_beta": 1.0,
                    "disturbance_weight": 0.1,
                    "diagnostics": {
                        "enabled": bool(self.pipeline.v3_config.smoke_test),
                        "filename": "runtime_diagnostics.jsonl",
                        "console": True,
                        "fsync": True,
                    },
                },
            },
        }
        return block

    def build_child_process_config(self) -> OrderedDict:
        child = copy.deepcopy(OrderedDict(self.pipeline.raw_process_config))
        for key in ("ideogram4_v3_activator", "three_phase_trigger_training"):
            child.pop(key, None)
        child["type"] = "sd_trainer"
        child["name"] = self.stage.name
        child["training_folder"] = self.pipeline.run_root
        child["network"] = None
        child["train"] = copy.deepcopy(child.get("train", {}))
        child["train"].update({
            "steps": self.stage.steps,
            "optimizer": self.stage.optimizer,
            "optimizer_params": copy.deepcopy(self.stage.optimizer_params),
            "train_unet": False,
            "train_text_encoder": False,
            "cache_text_embeddings": False,
            "unload_text_encoder": False,
            "disable_sampling": True,
        })
        if self.stage.name == "te_calibration":
            child["network"] = copy.deepcopy(self.pipeline.raw_process_config.get("v3_network", {"type": "lora", "linear": 32}))
            child["network"]["pretrained_lora_path"] = self.pipeline.v3_weights_path
        child["save"] = copy.deepcopy(child.get("save", {}))
        child["save"].update({
            "save_every": None,
            # BaseSDTrainProcess evaluates save_steps before incrementing step_num,
            # so completed update N is saved on loop index N - 1.
            "save_steps": [step - 1 for step in self.stage.save_steps],
        })
        child["sample"] = {"samples": []}
        child["trigger_selective_training"] = {
            "enabled": False,
            "caption_sources": self._caption_sources(),
        }
        child["three_phase_trigger_training"] = self._internal_three_phase()
        return child

    def build_child_job_config(self) -> OrderedDict:
        child = self.build_child_process_config()
        return OrderedDict({
            "job": "extension",
            "config": OrderedDict({"name": child["name"], "process": [child]}),
            "meta": copy.deepcopy(self.pipeline.job.meta),
        })

    def write_snapshot(self) -> str:
        payload = json.loads(json.dumps(self.build_child_job_config()))
        temp_path = self.snapshot_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        os.replace(temp_path, self.snapshot_path)
        return self.snapshot_path

    def execute(self) -> int:
        snapshot = self.write_snapshot()
        generic_save_root = os.path.join(self.pipeline.run_root, self.stage.name)
        internal_phase = "phase_a1" if self.stage.name == "semantic_activator" else "phase_a2"
        phase_output_root = os.path.join(self.pipeline.run_root, internal_phase)
        # The generic trainer resume format only contains the diffusion network and
        # optimizer, not the matching text activator components. A stage retry must
        # therefore be a clean attempt rather than a partial implicit resume, and
        # stale JSONL/artifacts from the failed attempt must not enter selection.
        for stale_root in (generic_save_root, phase_output_root):
            if os.path.isdir(stale_root):
                shutil.rmtree(stale_root)
        command = [sys.executable, os.path.join(TOOLKIT_ROOT, "run.py"), snapshot]
        return subprocess.run(command, cwd=TOOLKIT_ROOT, check=False).returncode
