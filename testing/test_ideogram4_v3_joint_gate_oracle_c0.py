from __future__ import annotations

import pytest
import torch

from extensions_built_in.ideogram4_v3_joint_gate_oracle_c0.config import load_config
from extensions_built_in.ideogram4_v3_joint_gate_oracle_c0.helpers import (
    build_prior_vectors,
    canonical_group_rows,
    interpolate_anchor_values,
    interpolation_weights,
    metric_payload,
    sign_agreement,
    select_balanced_batch,
    validate_visual_prompt_placeholders,
)


def _config(tmp_path, *, prompts=()):
    return {
        "a2_contract": "a2.json",
        "residual_manifest": "residual_manifest.json",
        "registry": "registry.json",
        "residual_records": "records.jsonl",
        "dataset_root": "dataset",
        "split_manifest": "split.json",
        "output_root": str(tmp_path / "out"),
        "timesteps": [100, 500, 900],
        "same_timestep_content_batch": 2,
        "optimizer": {"learning_rate": 0.01, "steps": 5, "noise_seeds": [42, 43, 44], "snapshot_steps": [0, 5]},
        "validation": {"noise_seeds": [314159]},
        "novel_visuals": {"prompts": list(prompts), "include_phase_c_v2": False},
    }


def test_c0_rejects_prompts_without_visual_seeds(tmp_path):
    raw = _config(tmp_path, prompts=("one",))
    raw["novel_visuals"]["seeds"] = []
    with pytest.raises(ValueError, match="seeds must be non-empty"):
        load_config(raw)


def test_visual_prompt_placeholder_contract_is_explicit():
    assert validate_visual_prompt_placeholders(("a [trigger] object",), "[trigger]")[0]["occurrence_count"] == 1
    with pytest.raises(ValueError, match=r"prompts\[0\].*at least once"):
        validate_visual_prompt_placeholders(("a plain object",), "[trigger]")
    with pytest.raises(ValueError, match="requires exactly 3"):
        validate_visual_prompt_placeholders(("[trigger] and [trigger]",), "[trigger]", expected_occurrences=3)
    audit = validate_visual_prompt_placeholders(
        ("[trigger] near [trigger] behind [trigger]",), "[trigger]", expected_occurrences=3
    )
    assert audit[0]["occurrence_count"] == 3


def test_c0_accepts_zero_one_or_many_novel_prompts(tmp_path):
    assert load_config(_config(tmp_path, prompts=())).novel_visuals.prompts == ()
    assert load_config(_config(tmp_path, prompts=("one",))).novel_visuals.prompts == ("one",)
    assert load_config(_config(tmp_path, prompts=("one", "two"))).novel_visuals.prompts == ("one", "two")


def test_c0_rejects_non_stage1_timesteps_and_q_bound_override(tmp_path):
    raw = _config(tmp_path)
    raw["timesteps"] = [100, 500]
    with pytest.raises(ValueError, match="exactly timesteps"):
        load_config(raw)
    raw = _config(tmp_path)
    raw["q_bound"] = 0.5
    with pytest.raises(ValueError, match="q_bound"):
        load_config(raw)


def test_c0_requires_safe_unique_artifact_paths(tmp_path):
    raw = _config(tmp_path)
    raw["artifacts"] = {"config": "../escape.json"}
    with pytest.raises(ValueError, match="artifact paths"):
        load_config(raw)


def test_endpoint_and_metric_contracts_are_exact():
    q = torch.zeros(102)
    q[0] = -0.25
    q[1] = 0.25
    assert 1.0 + q[0].item() == 0.75
    assert 1.0 + q[1].item() == 1.25
    assert metric_payload(2.0, 1.5)["absolute_gain"] == pytest.approx(0.5)
    assert sign_agreement(q[:2], torch.tensor([-0.25, 0.25]))["agreement_fraction"] == pytest.approx(1.0)


def test_three_anchor_interpolation_clamps_and_interpolates():
    assert interpolation_weights(50) == (0, 0, 0.0)
    assert interpolation_weights(900) == (2, 2, 0.0)
    assert interpolation_weights(300) == (0, 1, pytest.approx(0.5))
    assert interpolation_weights(700) == (1, 2, pytest.approx(0.5))
    anchors = {100: 1.0, 500: 2.0, 900: 4.0}
    assert interpolate_anchor_values(50, anchors) == pytest.approx(1.0)
    assert interpolate_anchor_values(100, anchors) == pytest.approx(1.0)
    assert interpolate_anchor_values(500, anchors) == pytest.approx(2.0)
    assert interpolate_anchor_values(900, anchors) == pytest.approx(4.0)
    assert interpolate_anchor_values(700, anchors) == pytest.approx(3.0)


def _group_rows():
    modules = []
    groups = {}
    for block in range(34):
        for family in ("adaln", "attention", "mlp"):
            group_id = f"block:{block}:{family}"
            modules.append({"module_name": f"module_{block}_{family}", "group_id": group_id, "block_index": block, "kind": family})
            groups[group_id] = [f"module_{block}_{family}"]
    return canonical_group_rows({"modules": modules, "groups": groups})


def test_registry_order_is_not_silently_rebuilt():
    rows = _group_rows()
    assert len(rows) == 102
    assert rows[0]["group_id"] == "block:0:adaln"
    assert rows[1]["group_id"] == "block:0:attention"
    assert rows[-1]["group_id"] == "block:33:mlp"


def test_fd_prior_keeps_missing_neutral_and_balances_train_heldout():
    rows = _group_rows()
    group = rows[0]["group_id"]
    records = []
    for row in rows:
        records.append({
            "run_id": "run", "probe_case_id": f"neutral-{row['group_id']}", "split": "train",
            "item_id": "neutral", "noise_seed": 42, "timestep": 100, "group_id": row["group_id"],
            "candidate_losses": {"1.0": 1.0},
            "metrics": {"dL_dg": 0.0, "finite_difference_selected": False, "fd_result": "not_selected_joint_gradient_only"},
        })
    for split, item, best, losses in (
        ("train", "a", 0.75, {"0.75": 0.8, "1.0": 1.0, "1.25": 1.2}),
        ("heldout", "b", 1.25, {"0.75": 1.2, "1.0": 1.0, "1.25": 0.8}),
    ):
        records.append({
            "run_id": "run", "probe_case_id": f"{split}-{item}", "split": split,
            "item_id": item, "noise_seed": 42, "timestep": 100, "group_id": group,
            "candidate_losses": losses,
            "metrics": {"dL_dg": 1.0 if split == "train" else -1.0, "finite_difference_selected": True, "fd_result": f"prefer_{best}", "best_scale": best, "left_slope": 1.0, "right_slope": 1.0, "central_secant": 1.0, "curvature": 1.0},
        })
    prior = build_prior_vectors(records, rows, 100)
    assert prior["details"][0]["fd_composition"] == pytest.approx((1.0, 0.0, 0.0))
    assert prior["fd_q"][0] == pytest.approx(-0.25)
    assert prior["fd_evidence_splits"] == ["train"]
    audit_prior = build_prior_vectors(records, rows, 100, fd_splits=("train", "heldout"))
    assert audit_prior["details"][0]["fd_composition"] == pytest.approx((0.5, 0.0, 0.5))
    assert audit_prior["fd_q"][0] == pytest.approx(0.0)
    assert prior["details"][1]["fd_status"] == "missing"
    assert prior["fd_q"][1] == pytest.approx(0.0)


def test_fd_prior_ignores_only_known_non_gate_diagnostic_groups():
    rows = _group_rows()
    records = []
    for row in rows:
        records.append({
            "run_id": "run", "probe_case_id": f"local-{row['group_id']}", "split": "train",
            "item_id": "item", "noise_seed": 42, "timestep": 100, "group_id": row["group_id"],
            "candidate_losses": {"1.0": 1.0},
            "metrics": {"dL_dg": 0.0, "finite_difference_selected": False, "fd_result": "not_selected_joint_gradient_only"},
        })
    for group in ("global:adaln", "global:other"):
        records.append({
            "run_id": "run", "probe_case_id": f"diagnostic-{group}", "split": "train",
            "item_id": "item", "noise_seed": 42, "timestep": 100, "group_id": group,
            "candidate_losses": {"1.0": 1.0}, "metrics": {"dL_dg": 1.0},
        })
    prior = build_prior_vectors(records, rows, 100)
    assert prior["ignored_non_gate_diagnostic_groups"] == ["global:adaln", "global:other"]
    assert prior["unknown_groups"] == []
    records.append({**records[-1], "probe_case_id": "bad", "group_id": "global:attention"})
    with pytest.raises(RuntimeError, match="unknown groups"):
        build_prior_vectors(records, rows, 100)


def test_fd_prior_rejects_truncated_group_coverage():
    rows = _group_rows()
    with pytest.raises(RuntimeError, match="lack all evidence"):
        build_prior_vectors([], rows, 100)


def test_train_batch_selection_has_distinct_items_when_dataset_not_divisible():
    items = [{"dataset_relative_item_id": str(index)} for index in range(10)]
    batches = [select_balanced_batch(items, step, 4, 42) for step in range(1, 8)]
    assert all(len({item["dataset_relative_item_id"] for item in batch}) == 4 for batch in batches)
    counts = {
        source["dataset_relative_item_id"]: sum(
            selected["dataset_relative_item_id"] == source["dataset_relative_item_id"]
            for batch in batches for selected in batch
        )
        for source in items
    }
    assert max(counts.values()) - min(counts.values()) <= 1


def test_fd_prior_symmetric_outer_tie_has_no_directional_bias():
    rows = _group_rows()
    group = rows[0]["group_id"]
    record = {
        "run_id": "run", "probe_case_id": "tie", "split": "train", "item_id": "a",
        "noise_seed": 42, "timestep": 100, "group_id": group,
        "candidate_losses": {"0.75": 0.8, "1.0": 1.0, "1.25": 0.8},
        "metrics": {"dL_dg": 0.0, "finite_difference_selected": True, "fd_result": "prefer_0.75", "best_scale": 0.75, "left_slope": 1.0, "right_slope": 1.0, "central_secant": 1.0, "curvature": 1.0},
    }
    neutral = []
    for row in rows[1:]:
        neutral.append({
            "run_id": "run", "probe_case_id": f"neutral-{row['group_id']}", "split": "train",
            "item_id": "neutral", "noise_seed": 42, "timestep": 100, "group_id": row["group_id"],
            "candidate_losses": {"1.0": 1.0},
            "metrics": {"dL_dg": 0.0, "finite_difference_selected": False, "fd_result": "not_selected_joint_gradient_only"},
        })
    prior = build_prior_vectors([record, *neutral], rows, 100)
    assert prior["details"][0]["fd_composition"] == pytest.approx((0.5, 0.0, 0.5))
    assert prior["fd_q"][0] == pytest.approx(0.0)
