"""Right-token truncation is explicit, suffix-aware, and fully auditable."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from extensions.gen2_trainer.text_preflight import (
    build_token_report, prepare_caption_tokens, require_token_budget,
    require_token_report, tokenize_caption,
)
from extensions.gen2_trainer.tests.test_text_preflight import (
    CaptionTokenizer, config_fixture, manifest_fixture,
)


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def truncate_config(limit=3072):
    config = config_fixture(limit=limit)
    config["gen2"]["conditioning"]["overflow_policy"] = "truncate"
    return config


def test_3233_tokens_retain_right_truncated_prefix_and_all_full_provenance():
    tokenizer = CaptionTokenizer({"long caption": 3233})
    tokenizer.truncation_side = "left"  # Gen2's explicit policy always retains the prefix.
    caption = "  [trigger] long caption <target>  "
    item = prepare_caption_tokens(caption, "<target>", tokenizer, 4, 3072, "truncate")
    assert item["original_ids"] == list(range(3068))
    assert item["original_length"] == 3068
    assert item["untruncated_length"] == 3233
    assert item["truncated_tokens"] == 165
    assert item["truncated"] and item["overflow"]
    assert item["overflow_policy"] == "truncate" and item["truncation_side"] == "right"
    assert item["untruncated_ids_sha256"] == ids_hash(list(range(3233)))
    assert item["encoded_ids_sha256"] == ids_hash(list(range(3068)))
    assert item["original_caption"] == caption
    assert item["q"] == "long caption"
    assert item["serialized_text"] == "<user>long caption</user><assistant>"
    assert tokenizer.truncation_side == "left"
    require_token_budget(item, 4, 3072)


@pytest.mark.parametrize("policy", ["error", "truncate"])
@pytest.mark.parametrize("length", [1, 100, 3068])
def test_short_and_exact_boundary_tokens_are_unchanged(policy, length):
    tokenizer = CaptionTokenizer({"caption": length})
    full = tokenize_caption("caption", "<target>", tokenizer)
    item = prepare_caption_tokens("caption", "<target>", tokenizer, 4, 3072, policy)
    for name in ("original_caption", "q", "serialized_text", "original_ids", "original_length"):
        assert item[name] == full[name]
    assert not item["truncated"] and not item["overflow"]
    assert item["truncated_tokens"] == 0 and item["untruncated_length"] == length
    assert item["encoded_ids_sha256"] == item["untruncated_ids_sha256"]


@pytest.mark.parametrize("reservation", [0, 1, 4, 32])
def test_reservation_limits_encoded_ids_independently_of_literal_trigger(reservation):
    tokenizer = CaptionTokenizer({"caption": 3233})
    neutral = prepare_caption_tokens("caption", "<target>", tokenizer, reservation, 3072, "truncate")
    styled = prepare_caption_tokens("[trigger] caption", "<target>", tokenizer, reservation, 3072, "truncate")
    assert neutral["original_ids"] == styled["original_ids"] == list(range(3072-reservation))
    assert neutral["original_length"] + reservation == 3072
    assert not neutral["trigger_present"] and styled["trigger_present"]


def test_default_error_policy_and_existing_helpers_still_reject_without_truncation():
    tokenizer = CaptionTokenizer({"caption": 3233})
    full = tokenize_caption("caption", "<target>", tokenizer)
    with pytest.raises(ValueError, match="original_length=3233, M=4, limit=3072.*overflow_policy=error"):
        prepare_caption_tokens("caption", "<target>", tokenizer, 4, 3072)
    with pytest.raises(ValueError, match="overflow_policy=error"):
        require_token_budget(full, 4, 3072)
    assert full["original_ids"] == list(range(3233))


def test_reports_accept_all_input_sources_and_preserve_raw_counts_and_hashes():
    config, manifest = truncate_config(), manifest_fixture()
    before = deepcopy((config, manifest))
    lengths = {"first": 3233, "second": 3068, "validation": 4000, "sample": 3072}
    report = build_token_report(config, manifest, CaptionTokenizer(lengths))
    assert report["passed"] and report["failures"] == []
    assert report["captions_checked"] == 4
    assert report["overflow_policy"] == "truncate"
    assert report["truncation_count"] == len(report["truncations"]) == 3
    assert [row["source"] for row in report["truncations"]] == ["training", "validation", "sampling"]
    assert report["max_original_length"] == 4000 and report["original_token_budget"] == 3068
    for row, length in zip(report["rows"], lengths.values()):
        encoded = min(length, 3068)
        assert row["original_length"] == length and row["total_length"] == length+4
        assert row["encoded_length"] == encoded and row["encoded_total_length"] == encoded+4
        assert row["truncated_tokens"] == length-encoded
        assert row["truncated"] == row["overflow"] == (length > 3068)
        assert row["original_ids_sha256"] == ids_hash(list(range(length)))
        assert row["encoded_ids_sha256"] == ids_hash(list(range(encoded)))
        assert row["overflow_policy"] == "truncate" and row["truncation_side"] == "right"
    assert (config, manifest) == before


def test_runtime_and_report_use_identical_prefix_hashes_for_every_source():
    config, manifest = truncate_config(), manifest_fixture()
    lengths = {"first": 3233, "second": 12, "validation": 3070, "sample": 4500}
    tokenizer = CaptionTokenizer(lengths)
    report = build_token_report(config, manifest, tokenizer)
    for row, caption in zip(report["rows"], lengths):
        prepared = prepare_caption_tokens(caption, "<target>", tokenizer, 4, 3072, "truncate")
        assert row["encoded_ids_sha256"] == prepared["encoded_ids_sha256"]
        assert row["original_ids_sha256"] == prepared["untruncated_ids_sha256"]
        assert row["encoded_length"] == prepared["original_length"]
        assert row["original_length"] == prepared["untruncated_length"]


def test_successful_truncation_logs_counts_sources_and_report_without_caption_text(capsys):
    config, manifest = truncate_config(), manifest_fixture()
    manifest[0]["original_caption"] = "PRIVATE FULL CAPTION CONTENT"
    tokenizer = CaptionTokenizer({"PRIVATE FULL CAPTION CONTENT": 3233, "validation": 3072, "sample": 4000})
    report = build_token_report(config, manifest, tokenizer)
    require_token_report(report, "/run/caption_token_report.json")
    output = capsys.readouterr().out
    for fragment in ("accepted 4 inputs", "overflow_policy=truncate", "right-truncated 3",
                     "training /dataset/first.png", "validation /dataset/validation.png", "sampling p000",
                     "full=3233 -> retained=3068", "discarded=165", "M=4", "limit=3072",
                     "/dataset/first.json", "sample.prompts[0]", "/run/caption_token_report.json"):
        assert fragment in output
    assert "PRIVATE FULL CAPTION CONTENT" not in output


def test_no_truncations_remains_silent_and_larger_override_does_not_mutate_config(capsys):
    config = truncate_config()
    report = build_token_report(config, manifest_fixture(), CaptionTokenizer({"first": 3233}), max_text_length=4096)
    assert report["passed"] and not report["truncations"] and report["truncation_count"] == 0
    require_token_report(report)
    assert capsys.readouterr().out == ""
    assert config["model"]["model_kwargs"]["max_text_length"] == 3072


@pytest.mark.parametrize("policy", ["error", "truncate"])
def test_empty_token_sequence_always_fails_and_report_keeps_other_rows(policy):
    tokenizer = CaptionTokenizer({"first": 0, "second": 3233, "validation": 3234, "sample": 3235})
    with pytest.raises(ValueError, match=f"overflow_policy={policy}"):
        prepare_caption_tokens("first", "<target>", tokenizer, 4, 3072, policy)
    config = truncate_config()
    config["gen2"]["conditioning"]["overflow_policy"] = policy
    report = build_token_report(config, manifest_fixture(), tokenizer)
    assert not report["passed"] and len(report["rows"]) == 4
    assert len(report["failures"]) == (4 if policy == "error" else 1)
    assert report["failures"][0]["empty_sequence"]
    with pytest.raises(ValueError, match=f"overflow_policy={policy}"):
        require_token_report(report)


@pytest.mark.parametrize("num_tokens,limit", [(-1, 3072), (4, 4), (5, 4), (0, 0),
                                            (4.0, 3072), (4, 3072.0), (True, 3072), (0, True)])
def test_invalid_reservations_fail_before_tokenizing(num_tokens, limit):
    tokenizer = CaptionTokenizer()
    with pytest.raises(ValueError, match="token reservation"):
        prepare_caption_tokens("caption", "<target>", tokenizer, num_tokens, limit, "truncate")
    assert tokenizer.calls == []


@pytest.mark.parametrize("policy", ["silent", "left", "", None])
def test_invalid_policy_fails_before_tokenizing_and_report_cannot_silently_accept(policy):
    tokenizer = CaptionTokenizer()
    with pytest.raises(ValueError, match="overflow policy"):
        prepare_caption_tokens("caption", "<target>", tokenizer, 4, 3072, policy)
    config = truncate_config()
    config["gen2"]["conditioning"]["overflow_policy"] = policy
    with pytest.raises(ValueError, match="overflow policy"):
        build_token_report(config, manifest_fixture(), tokenizer)
    assert tokenizer.calls == []


def test_prepared_ids_do_not_mutate_shared_tokenizer_result_or_caption_files(monkeypatch, tmp_path):
    shared = list(range(3233))

    class SharedTokenizer(CaptionTokenizer):
        def __call__(self, *args, **kwargs):
            super().__call__(*args, **kwargs)
            return {"input_ids": shared}

    def forbidden(*args, **kwargs):
        raise AssertionError("Truncation must not read or write caption files")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    item = prepare_caption_tokens("caption", "<target>", SharedTokenizer(), 4, 3072, "truncate")
    assert shared == list(range(3233))
    assert item["original_ids"] is not shared
    item["original_ids"][0] = -1
    assert shared[0] == 0
    assert not list(tmp_path.iterdir())


def test_json_digest_and_chat_serialization_stay_complete_even_when_ids_are_truncated():
    caption = (' { "high_level_description": "[trigger] café", '
               '"compositional_deconstruction": {"background": "blue", "elements": []} } ')
    tokenizer = CaptionTokenizer()
    full = tokenize_caption(caption, "<target>", tokenizer)
    prepared = prepare_caption_tokens(caption, "<target>", tokenizer, 4, 32, "truncate")
    assert prepared["truncated"] and prepared["original_ids"] == full["original_ids"][:28]
    assert prepared["serialized_text"] == full["serialized_text"]
    assert prepared["serialized_text"].endswith("}}</user><assistant>")
    assert prepared["q"] == full["q"] and prepared["original_caption"] == caption
