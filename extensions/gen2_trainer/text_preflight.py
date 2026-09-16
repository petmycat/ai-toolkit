"""Complete caption tokenization shared by preflight and the training encoder.

Counting tokens needs the native tokenizer, never the encoder or model weights.
No caption is shortened, skipped or rewritten beyond the native digest contract.
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


def require_token_budget(item, num_tokens, limit):
    budget = token_budget(item["original_length"], num_tokens, limit)
    if budget["overflow"] or budget["empty_sequence"]:
        raise ValueError(f"Gen2 token overflow: caption={item['q']!r}, original_length={item['original_length']}, "
                         f"M={num_tokens}, limit={limit}; over_by={budget['over_by']}; overflow_policy=error")


def build_token_report(config, manifest, tokenizer, max_text_length=None):
    """Scan every source, retaining all failures with file/prompt provenance."""
    num_tokens = config["gen2"]["conditioning"]["num_tokens"]
    limit = max_text_length if max_text_length is not None else config["model"]["model_kwargs"]["max_text_length"]
    rows = []

    def check(caption, **source):
        item = tokenize_caption(caption, config["trigger_word"], tokenizer)
        budget = token_budget(item["original_length"], num_tokens, limit)
        rows.append({**source, **budget,
                     "original_ids_sha256": hashlib.sha256(json.dumps(item["original_ids"]).encode()).hexdigest(),
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
    failures = [row for row in rows if row["overflow"] or row["empty_sequence"]]
    return {"passed": not failures, "overflow_policy": "error", "num_tokens": num_tokens,
            "limit": limit, "original_token_budget": limit-num_tokens,
            "captions_checked": len(rows), "max_original_length": max((row["original_length"] for row in rows), default=0),
            "rows": rows, "failures": failures}


def require_token_report(report, report_path=None):
    if report["passed"]:
        return
    details = []
    for row in report["failures"]:
        source = row.get("path", row.get("prompt_id", row["caption_provenance"]))
        details.append(f"{source} (caption: {row['caption_provenance']}): "
                       f"original_length={row['original_length']}, M={row['num_tokens']}, "
                       f"limit={row['limit']}, over_by={row['over_by']}, empty={row['empty_sequence']}")
    suffix = f" Full report: {report_path}." if report_path is not None else ""
    raise ValueError(f"Gen2 caption preflight rejected {len(details)} of {report['captions_checked']} inputs; "
                     f"overflow_policy=error, original token budget={report['original_token_budget']}."
                     f"{suffix}\n" + "\n".join(details))
