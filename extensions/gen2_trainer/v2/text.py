"""Native whole-prompt tokenization with explicit in-place soft-token markers."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import re


INTERNAL_MARKER = "<|gen2_v2_internal_activator_491c8f|>"
BOUNDARY_PROBE = "GEN2V2_CHAT_CONTENT_BOUNDARY_b26f8e"


@dataclass
class CompiledPrompt:
    ids: list[int]
    soft_positions: list[int]
    soft_bank_indices: list[int]
    metadata: dict

    @property
    def mode(self):
        return self.metadata["mode"]


class NativePromptCompiler:
    def __init__(self, tokenizer, trigger, num_tokens, max_length=3072, overflow_policy="truncate"):
        if not isinstance(trigger, str) or not trigger or num_tokens < 1 or max_length < 1:
            raise ValueError("A nonempty trigger, positive bank size and token budget are required")
        if overflow_policy not in ("truncate", "error"):
            raise ValueError("V2 overflow_policy must be truncate or error")
        self.tokenizer, self.trigger = tokenizer, trigger
        self.num_tokens, self.max_length, self.overflow_policy = num_tokens, max_length, overflow_policy
        self.marker_pattern = re.compile("|".join(re.escape(value) for value in sorted({"[trigger]", trigger}, key=len, reverse=True)))
        if INTERNAL_MARKER in tokenizer.get_vocab():
            raise ValueError("Reserved internal activator marker already exists in the original vocabulary")
        self.soft_tokenizer = copy.deepcopy(tokenizer)
        try:
            from tokenizers import AddedToken
            marker = AddedToken(INTERNAL_MARKER, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)
        except ImportError:
            # Lightweight tokenizer fixtures can implement the same reserved
            # special-token API; real Qwen installations require tokenizers.
            marker = INTERNAL_MARKER
        # Register only on the isolated tokenizer. add_tokens has the same API
        # in Transformers 4/5 and preserves native special-token attributes;
        # add_special_tokens renamed its replacement keyword in Transformers 5.
        # The private ID is expanded into safe IDs below, never sent to Qwen's
        # unchanged vocabulary embedding table.
        self.soft_tokenizer.add_tokens([marker], special_tokens=True)
        self.marker_id = self.soft_tokenizer.convert_tokens_to_ids(INTERNAL_MARKER)
        if not isinstance(self.marker_id, int) or self.marker_id in tokenizer.get_vocab().values():
            raise ValueError("Isolated tokenizer did not allocate a distinct internal marker ID")
        self.safe_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        probe = self._chat(BOUNDARY_PROBE)
        if probe.count(BOUNDARY_PROBE) != 1:
            raise ValueError("Native chat template does not expose one unambiguous text-content span")
        self.chat_prefix, self.chat_suffix = probe.split(BOUNDARY_PROBE)

    def _chat(self, content):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": content}]}],
            add_generation_prompt=True, tokenize=False)

    def _normalize(self, caption, require_trigger=False):
        from toolkit.ideogram_caption import digest_caption_string
        if INTERNAL_MARKER in caption or BOUNDARY_PROBE in caption:
            raise ValueError("Caption collides with Gen2's reserved internal compiler syntax")
        original_count = len(self.marker_pattern.findall(caption))
        if require_trigger and original_count == 0:
            raise ValueError("Training caption is missing [trigger] or the configured reserved trigger")
        normalized = digest_caption_string(caption)
        normalized_count = len(self.marker_pattern.findall(normalized))
        if original_count != normalized_count:
            raise ValueError(f"Native caption normalization changed trigger occurrences: source={original_count}, normalized={normalized_count}")
        return normalized, original_count

    def _render(self, normalized, mode, named_phrase):
        if mode not in ("learned", "init", "base", "named"):
            raise ValueError(f"Unknown v2 prompt mode: {mode}")
        if mode == "named" and (not isinstance(named_phrase, str) or not named_phrase):
            raise ValueError("Named comparison mode requires an evaluation-only named phrase")
        replacement = INTERNAL_MARKER if mode in ("learned", "init") else named_phrase if mode == "named" else ""
        matches = list(self.marker_pattern.finditer(normalized))
        pieces, cursor, replacement_spans, length = [], 0, [], 0
        for match in matches:
            ordinary = normalized[cursor:match.start()]
            pieces.extend((ordinary, replacement))
            length += len(ordinary)
            replacement_spans.append((length, length+len(replacement)))
            length += len(replacement)
            cursor = match.end()
        pieces.append(normalized[cursor:])
        return "".join(pieces), matches, replacement_spans

    def _tokenize(self, normalized, mode, named_phrase):
        rendered, matches, replacement_spans = self._render(normalized, mode, named_phrase)
        serialized = self._chat(rendered)
        if serialized != self.chat_prefix+rendered+self.chat_suffix:
            raise ValueError("Native chat-template boundaries changed with caption content")
        tokenizer = self.soft_tokenizer if mode in ("learned", "init") and matches else self.tokenizer
        try:
            encoded = tokenizer(serialized, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        except (TypeError, NotImplementedError) as error:
            raise ValueError("V2 preflight requires native fast-tokenizer offset mappings for protected truncation") from error
        ids, offsets = encoded["input_ids"], encoded.get("offset_mapping")
        if offsets is None or len(ids) != len(offsets):
            raise ValueError("Native tokenizer returned incomplete offset mapping")
        start, end = len(self.chat_prefix), len(self.chat_prefix)+len(rendered)
        expanded, soft, bank, spans, body_count, tail_boundaries = [], [], [], [], 0, []
        ordinary_positions, token_offsets = [], []
        last_source = matches[-1].end() if matches else 0
        last_rendered = replacement_spans[-1][1] if matches else 0
        seen = 0
        for token_id, offset in zip(ids, offsets):
            a, b = map(int, offset)
            token_position = len(expanded)
            is_marker = tokenizer is self.soft_tokenizer and int(token_id) == self.marker_id
            if is_marker:
                if a < start or b > end:
                    raise ValueError("Reserved marker appeared outside native caption content")
                position = len(expanded)
                expanded.extend([self.safe_id]*self.num_tokens)
                soft.extend(range(position, position+self.num_tokens))
                bank.extend(range(self.num_tokens))
                spans.append([position, position+self.num_tokens])
                seen += 1
            else:
                expanded.append(int(token_id))
            if a >= start and b <= end and b > a:
                body_count += self.num_tokens if is_marker else 1
                if not is_marker:
                    ordinary_positions.append(token_position)
                # Only the contiguous ordinary tail after the FINAL marker may
                # be removed. We never keep a late marker while deleting the
                # ordinary scene text preceding it.
                if not is_marker and a-start >= last_rendered:
                    tail_boundaries.append(last_source+(a-start-last_rendered))
            token_offsets.append((token_position, len(expanded), a-start, b-start))
        if mode in ("learned", "init") and seen != len(matches):
            raise ValueError(f"Tokenizer marker mismatch: normalized={len(matches)}, tokenized={seen}")
        occurrence_spans = []
        for a, b in replacement_spans:
            overlapping = [(first, last) for first, last, ta, tb in token_offsets if tb > a and ta < b]
            if overlapping:
                occurrence_spans.append([overlapping[0][0], overlapping[-1][1]])
            else:
                anchor = next((first for first, last, ta, tb in token_offsets if ta >= a), len(expanded))
                occurrence_spans.append([anchor, anchor])
        return {"ids": expanded, "soft": soft, "bank": bank, "spans": spans,
                "rendered": rendered, "serialized": serialized, "occurrences": len(matches),
                "body_count": body_count, "tail_boundaries": sorted(set(tail_boundaries)),
                "last_marker_end": last_source, "ordinary_positions": ordinary_positions,
                "occurrence_spans": occurrence_spans,
                "source_marker_spans": [[match.start(), match.end()] for match in matches]}

    def _compile_set(self, caption, modes, named_phrase, require_trigger):
        normalized, occurrences = self._normalize(caption, require_trigger)
        full = {mode: self._tokenize(normalized, mode, named_phrase) for mode in modes}
        if self.overflow_policy == "error":
            for mode, value in full.items():
                if len(value["ids"]) > self.max_length:
                    raise ValueError(f"Caption exceeds the explicit error-policy token budget: mode={mode}, "
                                     f"expanded_length={len(value['ids'])}, occurrences={occurrences}, "
                                     f"bank_size={self.num_tokens}, limit={self.max_length}")
        cutoff = len(normalized)
        current = full
        while any(len(value["ids"]) > self.max_length for value in current.values()):
            candidates = []
            for value in current.values():
                overflow = len(value["ids"])-self.max_length
                if overflow <= 0:
                    continue
                boundaries = [value for value in value["tail_boundaries"] if value < cutoff]
                if not boundaries:
                    raise ValueError(f"Cannot fit caption without removing a marker, earlier content, or chat boundaries: "
                                     f"expanded_length={len(value['ids'])}, occurrences={occurrences}, bank_size={self.num_tokens}, limit={self.max_length}")
                candidates.append(boundaries[max(0, len(boundaries)-overflow)])
            next_cutoff = min(candidates)
            if next_cutoff >= cutoff:
                raise RuntimeError("Protected token truncation made no progress")
            cutoff = next_cutoff
            current = {mode: self._tokenize(normalized[:cutoff], mode, named_phrase) for mode in modes}
            if any(value["occurrences"] != occurrences for value in current.values()):
                raise ValueError("Protected truncation would drop an activator occurrence")
        common = self.marker_pattern.sub("", normalized[:cutoff])
        common_hash = hashlib.sha256(common.encode("utf-8")).hexdigest()
        results = {}
        for mode, value in current.items():
            metadata = {"mode": mode, "original_caption": caption, "normalized_caption": normalized,
                "retained_normalized_caption": normalized[:cutoff], "resolved_caption": value["rendered"],
                "serialized_text": value["serialized"], "occurrence_count": occurrences,
                "insertion_spans": value["spans"], "soft_positions": value["soft"],
                "occurrence_token_spans": value["occurrence_spans"],
                "normalized_marker_character_spans": value["source_marker_spans"],
                "ordinary_caption_positions": value["ordinary_positions"],
                "soft_bank_indices": value["bank"], "num_vectors": self.num_tokens,
                "original_length": len(full[mode]["ids"]), "expanded_length": len(full[mode]["ids"]),
                "resulting_length": len(value["ids"]), "omitted_content_tokens": full[mode]["body_count"]-value["body_count"],
                "truncated": cutoff < len(normalized), "retained_character_count": cutoff,
                "common_ordinary_content": common, "common_ordinary_content_sha256": common_hash,
                "shared_comparison_content": len(modes) > 1, "overflow_policy": self.overflow_policy,
                "max_length": self.max_length, "placement": "in_place_shared_bank_each_occurrence",
                "positions": list(range(len(value["ids"]))), "input_ids": value["ids"],
                "named_phrase": named_phrase if mode == "named" else None}
            results[mode] = CompiledPrompt(value["ids"], value["soft"], value["bank"], metadata)
        return results

    def compile(self, caption, mode="learned", named_phrase=None, require_trigger=False):
        return self._compile_set(caption, [mode], named_phrase, require_trigger)[mode]

    def comparison(self, caption, named_phrase=None):
        modes = ["base", "init", "learned"]
        if named_phrase:
            modes.insert(1, "named")
        return self._compile_set(caption, modes, named_phrase, False)

    def report(self, records, prompts, named_phrase=None):
        entries, failures = [], []
        for index, record in enumerate(records):
            caption = record if isinstance(record, str) else record.get("original_caption", record.get("caption", record.get("q")))
            try:
                if not isinstance(caption, str):
                    raise ValueError("Training record has no text caption")
                item = self.compile(caption, require_trigger=True)
                entries.append({"kind": "training", "index": index, **item.metadata})
            except ValueError as error:
                failures.append({"kind": "training", "index": index, "caption": caption, "error": str(error)})
        for index, prompt in enumerate(prompts):
            caption = prompt if isinstance(prompt, str) else prompt.get("prompt", prompt.get("text"))
            try:
                for item in self.comparison(caption, named_phrase).values():
                    entries.append({"kind": "evaluation", "index": index, **item.metadata})
            except (TypeError, ValueError) as error:
                failures.append({"kind": "evaluation", "index": index, "caption": caption, "error": str(error)})
        return {"passed": not failures, "failures": failures, "entries": entries,
                "truncation_count": sum(entry["truncated"] for entry in entries),
                "max_expanded_length": max((entry["expanded_length"] for entry in entries), default=0),
                "max_resulting_length": max((entry["resulting_length"] for entry in entries), default=0)}
