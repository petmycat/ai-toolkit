"""Caption-budget failures must be complete, reproducible, and precede GPU work."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from extensions.gen2_trainer.config import resolve_process_config
from extensions.gen2_trainer.text_preflight import (
    build_token_report, require_token_budget, require_token_report, tokenize_caption,
)


class CaptionTokenizer:
    """A complete chat wrapper with configurable whole-sequence token counts."""
    name_or_path = "fixture-native-tokenizer"
    chat_template = "fixture-complete-chat"

    def __init__(self, lengths=None):
        self.lengths = lengths or {}
        self.calls = []

    def get_vocab(self):
        return {"fixture": 0}

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt and not tokenize
        content = messages[0]["content"][0]["text"]
        assert messages[0]["role"] == "user"
        self.calls.append(content)
        return "<user>" + content + "</user><assistant>"

    def __call__(self, text, *, add_special_tokens, truncation):
        # Reject any accidental return to native silent-truncation behavior.
        assert add_special_tokens is False and truncation is False
        assert text.startswith("<user>") and text.endswith("</user><assistant>")
        content = text[len("<user>"):-len("</user><assistant>")]
        length = self.lengths.get(content, len(text))
        return {"input_ids": list(range(length))}


def config_fixture(*, limit=2048, disabled=False):
    return resolve_process_config({
        "trigger_word": "<target>",
        "model": {"model_kwargs": {"max_text_length": limit}},
        "train": {"disable_sampling": disabled, "validation_config": {
            "validation_items": [{"image_path": "/dataset/validation.png", "prompt": "validation"}]}},
        "sample": {"prompts": ["sample"]},
        "gen2": {"conditioning": {"num_tokens": 4}},
    })


def manifest_fixture():
    return [{"sample_id": "training-one", "path": "/dataset/first.png",
             "caption_provenance": "/dataset/first.json", "dataset_index": 0,
             "original_caption": "first"},
            {"sample_id": "training-two", "path": "/dataset/second.png",
             "caption_provenance": "/dataset/captions.json", "dataset_index": 1,
             "original_caption": "second"}]


def test_complete_native_digest_keeps_json_and_chat_closings_and_removes_markers():
    caption = ('  { "high_level_description": "[trigger] café <target>", '
               '"compositional_deconstruction": {"background": "blue", "elements": []} }  ')
    tokenizer = CaptionTokenizer()
    item = tokenize_caption(caption, "<target>", tokenizer)
    expected = ('{"high_level_description":" café ",'
                '"compositional_deconstruction":{"background":"blue","elements":[]}}')
    assert tokenizer.calls == [expected]
    assert item["original_caption"] == caption
    assert "[trigger]" not in item["q"] and "<target>" not in item["q"]
    assert item["serialized_text"] == "<user>" + expected + "</user><assistant>"
    assert item["original_length"] == len(item["original_ids"])
    assert "café" in item["serialized_text"]


def test_plain_caption_internal_spacing_and_requested_text_are_preserved():
    caption = '  A  sign reads "DO NOT  ENTER" [trigger]  '
    tokenizer = CaptionTokenizer()
    item = tokenize_caption(caption, "<target>", tokenizer)
    assert item["q"] == 'A  sign reads "DO NOT  ENTER"'
    assert tokenizer.calls == [item["q"]]


@pytest.mark.parametrize("length,limit,passes", [
    (2044, 2048, True), (2045, 2048, False),
    (2055, 2048, False), (2055, 3072, True),
    (3068, 3072, True), (3069, 3072, False), (0, 3072, False),
])
def test_complete_sequence_boundary_including_reserved_suffix(length, limit, passes):
    item = tokenize_caption("caption", "<target>", CaptionTokenizer({"caption": length}))
    if passes:
        require_token_budget(item, num_tokens=4, limit=limit)
    else:
        with pytest.raises(ValueError, match=rf"original_length={length}, M=4, limit={limit}.*overflow_policy=error"):
            require_token_budget(item, num_tokens=4, limit=limit)
    # Failure must not repair, trim, or replace the token sequence.
    assert item["original_ids"] == list(range(length))


def test_report_collects_every_offender_with_source_provenance_and_keeps_inputs():
    config, manifest = config_fixture(), manifest_fixture()
    before = deepcopy((config, manifest))
    tokenizer = CaptionTokenizer({"first": 2055, "second": 2060, "validation": 2070, "sample": 2080})
    report = build_token_report(config, manifest, tokenizer)
    assert not report["passed"]
    assert report["captions_checked"] == len(report["failures"]) == 4
    assert report["max_original_length"] == 2080
    assert report["original_token_budget"] == 2044
    assert [row["source"] for row in report["failures"]] == ["training", "training", "validation", "sampling"]
    assert report["failures"][0]["path"] == "/dataset/first.png"
    assert report["failures"][1]["caption_provenance"] == "/dataset/captions.json"
    assert report["failures"][2]["caption_provenance"] == "train.validation_config.validation_items[0].prompt"
    assert report["failures"][3]["prompt_id"] == "p000"
    assert (config, manifest) == before
    with pytest.raises(ValueError) as failure:
        require_token_report(report, "/outputs/caption_token_report.json")
    message = str(failure.value)
    for fragment in ("rejected 4 of 4", "/dataset/first.png", "/dataset/captions.json",
                     "/dataset/validation.png", "p000", "sample.prompts[0]",
                     "over_by=11", "overflow_policy=error", "/outputs/caption_token_report.json"):
        assert fragment in message


def test_empty_sequence_is_reported_even_without_numeric_overflow():
    config = config_fixture()
    report = build_token_report(config, manifest_fixture(), CaptionTokenizer({"first": 0}))
    assert not report["passed"]
    assert len(report["failures"]) == 1
    failure = report["failures"][0]
    assert failure["empty_sequence"] and not failure["overflow"]
    assert failure["over_by"] == 0
    with pytest.raises(ValueError, match="empty=True"):
        require_token_report(report)


def test_disabling_sampling_excludes_only_sampling_inputs():
    report = build_token_report(config_fixture(disabled=True), manifest_fixture(),
                                CaptionTokenizer({"validation": 2055, "sample": 5000}))
    assert report["captions_checked"] == 3
    assert [row["source"] for row in report["failures"]] == ["validation"]
    assert all(row["source"] != "sampling" for row in report["rows"])


def test_validation_item_without_prompt_checks_complete_empty_caption_chat():
    config = config_fixture()
    del config["train"]["validation_config"]["validation_items"][0]["prompt"]
    tokenizer = CaptionTokenizer()
    report = build_token_report(config, manifest_fixture(), tokenizer)
    assert report["passed"] and report["captions_checked"] == 4
    validation = next(row for row in report["rows"] if row["source"] == "validation")
    assert tokenizer.calls == ["first", "second", "", "sample"]
    assert validation["original_length"] == len("<user></user><assistant>")
    assert validation["total_length"] == validation["original_length"] + 4
    assert not validation["empty_sequence"]
    assert validation["path"] == "/dataset/validation.png"
    assert validation["caption_provenance"] == "train.validation_config.validation_items[0].prompt"


def test_larger_limit_preserves_all_original_sequences_and_hashes():
    config = config_fixture()
    tokenizer = CaptionTokenizer({"first": 2055, "second": 3068})
    old = build_token_report(config, manifest_fixture(), tokenizer)
    new = build_token_report(config, manifest_fixture(), tokenizer, max_text_length=3072)
    assert not old["passed"] and new["passed"]
    require_token_report(new)
    for before, after in zip(old["rows"], new["rows"]):
        for field in ("original_length", "total_length", "original_ids_sha256", "serialized_text_sha256"):
            assert before[field] == after[field]
    assert config["model"]["model_kwargs"]["max_text_length"] == 2048


def test_report_reads_caption_data_without_opening_dataset_or_writing_files(monkeypatch, tmp_path):
    # The manifest is authoritative; token checks need no image/caption reads.
    def forbidden(*args, **kwargs):
        raise AssertionError("Token reporting must not mutate or reopen dataset files")
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    report = build_token_report(config_fixture(), manifest_fixture(), CaptionTokenizer())
    assert report["passed"]
    assert not list(tmp_path.iterdir())


def test_startup_rejects_caption_before_model_construction_and_latent_cache(monkeypatch, tmp_path):
    """Exercise the real startup method, replacing only external GPU/services."""
    import numpy as np
    import extensions.gen2_trainer.process as process

    model_factory = Mock(side_effect=AssertionError("Model weights must not load before caption acceptance"))
    module_values = {
        "toolkit.accelerator": {"get_accelerator": lambda: SimpleNamespace(num_processes=1, scaler=None)},
        "toolkit.logging_aitk": {"create_logger": Mock()},
        "extensions_built_in.diffusion_models.ideogram4.ideogram4": {"Ideogram4Model": model_factory},
    }
    for name, attributes in module_values.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(process, "native_configuration", lambda _: {"logging": object(), "model": object()})
    monkeypatch.setattr(process, "Recorder", Mock())
    monkeypatch.setattr(process, "preflight_datasets", lambda _: manifest_fixture())
    loader = Mock(side_effect=AssertionError("Latents must not cache before caption acceptance"))
    monkeypatch.setattr(process, "make_native_loader", loader)
    monkeypatch.setattr(process.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(process.torch, "manual_seed", Mock())
    monkeypatch.setattr(process.torch, "use_deterministic_algorithms", Mock())
    monkeypatch.setattr(process.random, "seed", Mock())
    monkeypatch.setattr(np.random, "seed", Mock())
    monkeypatch.setattr("extensions.gen2_trainer.text_preflight.load_tokenizer",
                        lambda _: CaptionTokenizer({"first": 2055, "sample": 2060}))
    config = config_fixture()
    config["training_folder"] = str(tmp_path)
    runner = process.Gen2Runner(config, "overflow")
    with pytest.raises(ValueError, match="caption preflight rejected 2 of 4"):
        runner._load()
    model_factory.assert_not_called()
    loader.assert_not_called()
    report = json.loads((runner.root / "caption_token_report.json").read_text(encoding="utf-8"))
    assert not report["passed"]
    assert {row["source"] for row in report["failures"]} == {"training", "sampling"}
    assert runner.engine is runner.backend is runner.loader is None
