"""Caption tokenization and explicit overflow policy shared by every encoder.

Counting tokens needs the native tokenizer, never the encoder or model weights.
Full caption strings stay intact. Optional truncation changes only encoded IDs.
"""
from __future__ import annotations

import hashlib
import json
import os


def load_tokenizer(config, *, local_files_only=False):
    from transformers import AutoTokenizer
    path = config["model"].get("model_kwargs", {}).get("text_encoder_path", "Qwen/Qwen3-VL-8B-Instruct")
    return AutoTokenizer.from_pretrained(path, token=os.getenv("HF_TOKEN"), local_files_only=local_files_only)


def tokenize_caption(caption, trigger_word, tokenizer):
    from toolkit.ideogram_caption import digest_caption_string
    from .conditioning import compile_trigger
    item = compile_trigger(caption, trigger_word)
    serialized = tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": digest_caption_string(item["q"])}]}],
        add_generation_prompt=True, tokenize=False)
    ids = tokenizer(serialized, add_special_tokens=False, truncation=False)["input_ids"]
    item.update(serialized_text=serialized, original_ids=ids, original_length=len(ids))
    return item


def token_budget(original_length, num_tokens, limit):
    total = original_length + num_tokens
    return {"original_length": original_length, "num_tokens": num_tokens,
            "total_length": total, "limit": limit, "over_by": max(0, total-limit),
            "overflow": total > limit, "empty_sequence": original_length == 0}


def require_token_budget(item, num_tokens, limit, overflow_policy="error"):
    budget = token_budget(item["original_length"], num_tokens, limit)
    if budget["overflow"] or budget["empty_sequence"]:
        raise ValueError(f"Gen2 token overflow: caption={item['q']!r}, original_length={item['original_length']}, "
                         f"M={num_tokens}, limit={limit}; over_by={budget['over_by']}; overflow_policy={overflow_policy}")


def _validate_policy(num_tokens, limit, overflow_policy):
    if overflow_policy not in ("error", "truncate"):
        raise ValueError(f"Unknown Gen2 caption overflow policy: {overflow_policy!r}")
    if (not isinstance(num_tokens, int) or isinstance(num_tokens, bool) or num_tokens < 0 or
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or num_tokens >= limit):
        raise ValueError("Gen2 token reservation requires integer M >=0 and limit > M, leaving at least one caption token")


def _ids_hash(ids):
    # Preserve the historical report digest encoding for original_ids_sha256.
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def _prepare_token_item(item, num_tokens, limit, overflow_policy):
    """Apply policy without throwing for individual overflows or empty inputs.

    Preflight uses this form to collect all failures; runtime checks the result
    before encoding. The incoming tokenization item and ID list are untouched.
    """
    full_ids = item["original_ids"]
    full_length = item["original_length"]
    encoded_ids = list(full_ids[:limit-num_tokens] if overflow_policy == "truncate" else full_ids)
    dropped = full_length - len(encoded_ids)
    return {**item, "original_ids": encoded_ids, "original_length": len(encoded_ids),
            "untruncated_length": full_length, "untruncated_ids_sha256": _ids_hash(full_ids),
            "encoded_ids_sha256": _ids_hash(encoded_ids), "truncated_tokens": dropped,
            "truncated": dropped > 0, "overflow_policy": overflow_policy,
            "truncation_side": "right", "overflow": full_length + num_tokens > limit}


def prepare_caption_tokens(caption, trigger_word, tokenizer, num_tokens, limit, overflow_policy="error"):
    """Tokenize completely, then optionally retain the prefix fitting beside M.

    ``original_ids`` and ``original_length`` describe the prefix actually sent
    to the encoder. Full strings, the untruncated length, and both ID hashes
    retain provenance. Callers reserve the same M for styled and neutral paths.
    """
    _validate_policy(num_tokens, limit, overflow_policy)
    item = _prepare_token_item(tokenize_caption(caption, trigger_word, tokenizer),
                               num_tokens, limit, overflow_policy)
    require_token_budget(item, num_tokens, limit, overflow_policy)
    return item


def build_token_report(config, manifest, tokenizer, max_text_length=None):
    """Scan every source, retaining all failures with file/prompt provenance."""
    num_tokens = config["gen2"]["conditioning"]["num_tokens"]
    overflow_policy = config["gen2"]["conditioning"].get("overflow_policy", "error")
    limit = max_text_length if max_text_length is not None else config["model"]["model_kwargs"]["max_text_length"]
    _validate_policy(num_tokens, limit, overflow_policy)
    rows = []

    def check(caption, **source):
        item = _prepare_token_item(tokenize_caption(caption, config["trigger_word"], tokenizer),
                                   num_tokens, limit, overflow_policy)
        budget = token_budget(item["untruncated_length"], num_tokens, limit)
        rows.append({**source, **budget,
                     "original_ids_sha256": item["untruncated_ids_sha256"],
                     "encoded_ids_sha256": item["encoded_ids_sha256"],
                     "encoded_length": item["original_length"],
                     "encoded_total_length": item["original_length"] + num_tokens,
                     "truncated_tokens": item["truncated_tokens"], "truncated": item["truncated"],
                     "overflow_policy": overflow_policy, "truncation_side": "right",
                     "serialized_text_sha256": hashlib.sha256(item["serialized_text"].encode()).hexdigest()})

    for row in manifest:
        check(row["original_caption"], source="training", sample_id=row["sample_id"],
              path=row["path"], caption_provenance=row["caption_provenance"], dataset_index=row["dataset_index"])
    validation = config["train"].get("validation_config") or {}
    for index, item in enumerate(validation.get("validation_items", [])):
        check(item.get("prompt", ""), source="validation", path=item["image_path"],
              caption_provenance=f"train.validation_config.validation_items[{index}].prompt")
    if not config["train"]["disable_sampling"]:
        for index, prompt in enumerate(config["sample"]["prompts"]):
            check(prompt, source="sampling", prompt_id=f"p{index:03d}",
                  caption_provenance=f"sample.prompts[{index}]")
    failures = [row for row in rows if row["encoded_total_length"] > limit or row["empty_sequence"]]
    truncations = [row for row in rows if row["truncated"]]
    return {"passed": not failures, "overflow_policy": overflow_policy, "num_tokens": num_tokens,
            "limit": limit, "original_token_budget": limit-num_tokens,
            "captions_checked": len(rows), "max_original_length": max((row["original_length"] for row in rows), default=0),
            "rows": rows, "failures": failures, "truncations": truncations, "truncation_count": len(truncations)}


def require_token_report(report, report_path=None):
    suffix = f" Full report: {report_path}." if report_path is not None else ""
    if report["passed"]:
        truncations = report.get("truncations", [])
        if truncations:
            print(f"Gen2 caption preflight accepted {report['captions_checked']} inputs; "
                  f"overflow_policy={report['overflow_policy']}; right-truncated {len(truncations)} caption(s); "
                  f"original token budget={report['original_token_budget']}.{suffix}", flush=True)
            for row in truncations:
                source = row.get("path", row.get("prompt_id", row["caption_provenance"]))
                print(f"  {row['source']} {source} (caption: {row['caption_provenance']}): "
                      f"full={row['original_length']} -> retained={row['encoded_length']}, "
                      f"discarded={row['truncated_tokens']}, M={row['num_tokens']}, limit={row['limit']}", flush=True)
        return
    details = []
    for row in report["failures"]:
        source = row.get("path", row.get("prompt_id", row["caption_provenance"]))
        details.append(f"{source} (caption: {row['caption_provenance']}): "
                       f"original_length={row['original_length']}, M={row['num_tokens']}, "
                       f"limit={row['limit']}, over_by={row['over_by']}, empty={row['empty_sequence']}")
    raise ValueError(f"Gen2 caption preflight rejected {len(details)} of {report['captions_checked']} inputs; "
                     f"overflow_policy={report['overflow_policy']}, original token budget={report['original_token_budget']}."
                     f"{suffix}\n" + "\n".join(details))
