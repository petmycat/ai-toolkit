from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from toolkit.paths import get_path

from .config import STAGE_NAMES
from .stage_process import Ideogram4V3ActivatorStageProcess


class Ideogram4V3ActivatorPipeline:
    def initialize_pipeline(self) -> None:
        configured_root = self.v3_config.output_root or os.path.join(
            get_path(self.get_conf("training_folder", required=True)), self.name
        )
        self.run_root = get_path(configured_root)
        self.snapshot_root = os.path.join(self.run_root, "stage_configs")
        self.contract_root = os.path.join(self.run_root, "contracts")
        os.makedirs(self.snapshot_root, exist_ok=True)
        os.makedirs(self.contract_root, exist_ok=True)
        self.v3_weights_path = get_path(self.v3_config.v3_weights)
        if not os.path.isfile(self.v3_weights_path):
            raise FileNotFoundError(f"V3 snapshot/weights not found: {self.v3_weights_path}")
        self.literal = self._resolve_literal()
        self.stages = {
            "semantic_activator": Ideogram4V3ActivatorStageProcess(self, self.v3_config.semantic_activator),
            "te_calibration": Ideogram4V3ActivatorStageProcess(self, self.v3_config.te_calibration),
        }

    @staticmethod
    def sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _resolve_literal(self) -> str:
        explicit = self.get_conf("trigger_word", None)
        if explicit and str(explicit).strip():
            return str(explicit).strip()
        fallback = self.v3_config.literal_initialization.get("fallback_literal")
        if fallback and str(fallback).strip():
            return str(fallback).strip()
        raise RuntimeError(
            "configure trigger_word or literal_initialization.fallback_literal; the local "
            "literal_initialization module maps frozen Qwen rows and does not invent literals"
        )

    def a1_artifact_paths(self) -> Dict[str, str]:
        artifacts = self.v3_config.semantic_activator.artifacts
        root = os.path.join(self.run_root, "phase_a1", artifacts.final_dir)
        return {
            "embedding": os.path.join(root, artifacts.embedding_filename),
            "te_adapter": os.path.join(root, artifacts.te_adapter_filename),
            "literal_initialization": os.path.join(
                self.run_root,
                "phase_a1",
                str(self.v3_config.literal_initialization.get(
                    "artifact_filename", "literal_initialization.safetensors"
                )),
            ),
            "literal_initialization_manifest": os.path.join(
                self.run_root,
                "phase_a1",
                str(self.v3_config.literal_initialization.get(
                    "manifest_filename", "literal_initialization_manifest.json"
                )),
            ),
        }

    def _stage_sources(self, stage_name: str) -> Dict[str, Optional[str]]:
        if stage_name == "semantic_activator":
            return {"v3_weights": None, "a1_embedding": None, "a1_te_adapter": None}
        a1 = self.a1_artifact_paths()
        return {
            "v3_weights": self.v3_weights_path,
            "a1_embedding": a1["embedding"],
            "a1_te_adapter": a1["te_adapter"],
        }

    def _source_records(self, stage_name: str) -> Dict[str, Dict[str, Optional[str]]]:
        return {
            name: {
                "path": path,
                "sha256": self.sha256_file(path) if path and os.path.isfile(path) else None,
            }
            for name, path in self._stage_sources(stage_name).items()
        }

    def _write_json_atomic(self, path: str, payload: Dict[str, Any]) -> None:
        temp_path = path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        os.replace(temp_path, path)

    def _artifact_records(self, stage_name: str) -> Dict[str, str]:
        stage = self.stages[stage_name].stage
        internal_phase = "phase_a1" if stage_name == "semantic_activator" else "phase_a2"
        root = os.path.join(self.run_root, internal_phase)
        final = os.path.join(root, stage.artifacts.final_dir)
        records = {
            "metrics": os.path.join(root, stage.artifacts.metrics_file),
            "heldout_validation": os.path.join(root, stage.artifacts.validation_file),
            "final_dir": final,
            "embedding": os.path.join(final, stage.artifacts.embedding_filename),
            "te_adapter": os.path.join(final, stage.artifacts.te_adapter_filename),
        }
        if stage_name == "te_calibration":
            best = os.path.join(root, stage.artifacts.best_dir)
            records.update({
                "best_dir": best,
                "best_manifest": os.path.join(best, "best_heldout.json"),
                "best_embedding": os.path.join(best, stage.artifacts.embedding_filename),
                "best_te_adapter": os.path.join(best, stage.artifacts.te_adapter_filename),
            })
        return records

    def completion_contract(self, stage_name: str, status: str, return_code: Optional[int]) -> Dict[str, Any]:
        stage = self.stages[stage_name]
        snapshot_hash = self.sha256_file(stage.snapshot_path) if os.path.isfile(stage.snapshot_path) else None
        return {
            "schema": "ai-toolkit.ideogram4-v3-activator-stage-contract",
            "schema_version": 1,
            "pipeline": "ideogram4_v3_activator",
            "stage": stage_name,
            "automatic_order": list(STAGE_NAMES) + ["STOP"],
            "status": status,
            "return_code": return_code,
            "completed_at": datetime.now(timezone.utc).isoformat() if status in {"completed", "failed"} else None,
            "config_snapshot": stage.snapshot_path,
            "config_snapshot_sha256": snapshot_hash,
            "sources": self._source_records(stage_name),
            "source_fingerprint": hashlib.sha256(
                json.dumps(self._source_records(stage_name), sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "artifacts": self._artifact_records(stage_name),
            "training": {
                "steps": stage.stage.steps,
                "fresh_optimizer": True,
                "network": "disabled" if stage_name == "semantic_activator" else "V3 active/frozen",
                "embedding": "train" if stage_name == "semantic_activator" else "frozen",
                "text_encoder_adapter": "train",
                "taps": "disabled",
            },
        }

    def write_contract(self, stage_name: str, status: str, return_code: Optional[int] = None) -> str:
        path = self.stages[stage_name].contract_path
        self._write_json_atomic(path, self.completion_contract(stage_name, status, return_code))
        return path

    def _verify_inputs(self, stage_name: str) -> None:
        missing = [record["path"] for record in self._source_records(stage_name).values() if record["path"] and not record["sha256"]]
        if missing:
            raise FileNotFoundError("missing stage input artifact(s): " + "; ".join(missing))

    def _select_best_a2(self) -> None:
        artifacts = self._artifact_records("te_calibration")
        validation_path = artifacts["heldout_validation"]
        candidates = []
        if os.path.isfile(validation_path):
            with open(validation_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    metric = next((record.get(key) for key in ("heldout_loss", "target_mse", "loss") if isinstance(record.get(key), (int, float))), None)
                    record_step = int(record.get("step", 0))
                    if metric is not None and record_step > 0:
                        candidates.append((float(metric), record_step))
        final_files = [artifacts["embedding"], artifacts["te_adapter"]]
        if not candidates:
            raise RuntimeError("A2 completion requires heldout validation records containing heldout_loss, target_mse, or loss")
        best_loss, best_step = min(candidates)
        checkpoint_dir = os.path.join(self.run_root, "phase_a2", self.v3_config.te_calibration.artifacts.checkpoint_dir, str(best_step))
        sources = [
            os.path.join(checkpoint_dir, self.v3_config.te_calibration.artifacts.embedding_filename),
            os.path.join(checkpoint_dir, self.v3_config.te_calibration.artifacts.te_adapter_filename),
        ]
        if not all(os.path.isfile(path) for path in sources):
            if best_step == self.v3_config.te_calibration.steps and all(os.path.isfile(path) for path in final_files):
                sources = final_files
            else:
                raise FileNotFoundError(f"best heldout checkpoint artifacts missing for step {best_step}: {sources}")
        os.makedirs(artifacts["best_dir"], exist_ok=True)
        destinations = [artifacts["best_embedding"], artifacts["best_te_adapter"]]
        for source, destination in zip(sources, destinations):
            temp_path = destination + ".tmp"
            shutil.copyfile(source, temp_path)
            os.replace(temp_path, destination)
        self._write_json_atomic(artifacts["best_manifest"], {
            "schema": "ai-toolkit.ideogram4-v3-activator-best-heldout",
            "schema_version": 1,
            "selection": "minimum heldout loss",
            "step": best_step,
            "heldout_loss": best_loss,
            "artifacts": {
                "embedding": {"path": destinations[0], "sha256": self.sha256_file(destinations[0])},
                "te_adapter": {"path": destinations[1], "sha256": self.sha256_file(destinations[1])},
            },
        })

    def _verify_outputs(self, stage_name: str) -> None:
        artifacts = self._artifact_records(stage_name)
        required = [artifacts["metrics"], artifacts["embedding"], artifacts["te_adapter"]]
        if stage_name == "semantic_activator":
            a1 = self.a1_artifact_paths()
            required.extend([a1["literal_initialization"], a1["literal_initialization_manifest"]])
        if stage_name == "te_calibration":
            self._select_best_a2()
            required.extend([artifacts["heldout_validation"], artifacts["best_manifest"], artifacts["best_embedding"], artifacts["best_te_adapter"]])
        missing = [path for path in required if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError("stage completion artifacts missing: " + "; ".join(missing))

    def contract_verified(self, stage_name: str) -> bool:
        path = self.stages[stage_name].contract_path
        if not os.path.isfile(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as handle:
                contract = json.load(handle)
            if contract.get("status") != "completed" or contract.get("return_code") != 0:
                return False
            if contract.get("config_snapshot_sha256") != self.sha256_file(contract["config_snapshot"]):
                return False
            if contract.get("sources") != self._source_records(stage_name):
                return False
            self._verify_outputs(stage_name)
            return True
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return False

    def run_pipeline(self) -> None:
        for stage_name in STAGE_NAMES:
            if self.contract_verified(stage_name):
                continue
            self._verify_inputs(stage_name)
            stage = self.stages[stage_name]
            stage.write_snapshot()
            self.write_contract(stage_name, "running")
            return_code = stage.execute()
            status = "completed" if return_code == 0 else "failed"
            if return_code == 0:
                try:
                    self._verify_outputs(stage_name)
                except Exception:
                    status = "failed"
                    return_code = 1
            self.write_contract(stage_name, status, return_code)
            if return_code != 0:
                raise RuntimeError(f"Ideogram4 V3 activator stage {stage_name} failed with exit code {return_code}")
        self._write_json_atomic(os.path.join(self.run_root, "pipeline_completion.json"), {
            "schema": "ai-toolkit.ideogram4-v3-activator-pipeline",
            "schema_version": 1,
            "status": "completed",
            "automatic_order": ["semantic_activator", "te_calibration", "STOP"],
            "terminal_manual_ablation": self.v3_config.terminal_manual_ablation,
            "contracts": {name: self.stages[name].contract_path for name in STAGE_NAMES},
        })
