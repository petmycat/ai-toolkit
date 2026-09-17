import copy
import unittest

from extensions.gen2_trainer.v2.text import NativePromptCompiler, INTERNAL_MARKER


class FixtureTokenizer:
    """Offset-aware whole-string tokenizer fixture, including one merge rule."""
    def __init__(self):
        self.vocabulary = {chr(index): index for index in range(128)}
        self.special = ["<begin>", "<end>", "<assistant>"]
        for token in [*self.special, "xy"]:
            self.vocabulary[token] = len(self.vocabulary)
        self.eos_token_id = self.vocabulary["<end>"]
        self.all_special_ids = [self.vocabulary[token] for token in self.special]

    def get_vocab(self):
        return dict(self.vocabulary)

    def add_special_tokens(self, mapping, replace_additional_special_tokens=False):
        for value in mapping["additional_special_tokens"]:
            token = str(value)
            self.vocabulary[token] = len(self.vocabulary)
            self.special.append(token)
            self.all_special_ids.append(self.vocabulary[token])

    def convert_tokens_to_ids(self, token):
        return self.vocabulary.get(token)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        assert add_generation_prompt and not tokenize
        return "<begin>user\n"+messages[0]["content"][0]["text"]+"<end>\n<assistant>"

    def __call__(self, text, add_special_tokens=False, truncation=False, return_offsets_mapping=False):
        assert not add_special_tokens and not truncation
        ids, offsets, index = [], [], 0
        while index < len(text):
            token = next((candidate for candidate in sorted([*self.special, "xy"], key=len, reverse=True)
                          if text.startswith(candidate, index)), text[index])
            ids.append(self.vocabulary[token]); offsets.append((index, index+len(token))); index += len(token)
        return {"input_ids": ids, **({"offset_mapping": offsets} if return_offsets_mapping else {})}


class V2TextTests(unittest.TestCase):
    def test_repeated_markers_share_ordered_bank_and_original_tokenizer(self):
        tokenizer = FixtureTokenizer(); original = tokenizer.get_vocab()
        compiler = NativePromptCompiler(tokenizer, "<s>", 4, 3072)
        item = compiler.compile("A [trigger] tiger <s> room [trigger] sky", require_trigger=True)
        self.assertEqual(item.metadata["occurrence_count"], 3)
        self.assertEqual(len(item.soft_positions), 12)
        self.assertEqual(item.soft_bank_indices, list(range(4))*3)
        self.assertEqual(item.metadata["insertion_spans"], [[p, p+4] for p in item.soft_positions[::4]])
        self.assertEqual(tokenizer.get_vocab(), original)
        self.assertNotIn(INTERNAL_MARKER, tokenizer.get_vocab())

    def test_base_uses_whole_prompt_native_tokenization(self):
        tokenizer = FixtureTokenizer(); compiler = NativePromptCompiler(tokenizer, "<s>", 4)
        item = compiler.compile("x[trigger]y", mode="base")
        native = tokenizer(tokenizer.apply_chat_template([{"content": [{"text": "xy"}]}]))["input_ids"]
        self.assertEqual(item.ids, native)
        self.assertIn(tokenizer.vocabulary["xy"], item.ids)
        self.assertFalse(item.soft_positions)
        no_marker = compiler.compile("plain xy text")
        self.assertEqual(no_marker.ids, tokenizer(no_marker.metadata["serialized_text"])["input_ids"])

    def test_native_json_order_and_marker_loss_preflight(self):
        compiler = NativePromptCompiler(FixtureTokenizer(), "<s>", 4)
        caption = '{"style_description":{"art_style":"[trigger]","medium":"[trigger]","aesthetics":"[trigger]"},"high_level_description":"A tiger","compositional_deconstruction":{}}'
        item = compiler.compile(caption, require_trigger=True)
        normalized = item.metadata["normalized_caption"]
        self.assertLess(normalized.index("high_level_description"), normalized.index("style_description"))
        self.assertEqual(item.metadata["occurrence_count"], 3)
        with self.assertRaisesRegex(ValueError, "normalization changed"):
            compiler.compile('{"aspect_ratio":"[trigger]","high_level_description":"A tiger","compositional_deconstruction":{}}', require_trigger=True)

    def test_explicit_overflow_error_reports_expansion_before_truncation(self):
        compiler = NativePromptCompiler(FixtureTokenizer(), "<s>", 4, 24, "error")
        with self.assertRaisesRegex(ValueError, "error-policy.*expanded_length=.*occurrences=1.*limit=24"):
            compiler.compile("[trigger] " + "ordinary caption "*5, require_trigger=True)
        short = compiler.compile("[trigger]")
        self.assertFalse(short.metadata["truncated"])
        self.assertEqual(short.metadata["overflow_policy"], "error")

    def test_shared_comparison_truncates_only_contiguous_ordinary_tail(self):
        compiler = NativePromptCompiler(FixtureTokenizer(), "<s>", 4, 42)
        caption = "A [trigger] tiger with " + "a long detailed room "*5
        comparison = compiler.comparison(caption, "LONG NAMED PHRASE")
        self.assertEqual(set(comparison), {"base", "named", "init", "learned"})
        self.assertEqual(len({item.metadata["common_ordinary_content_sha256"] for item in comparison.values()}), 1)
        self.assertEqual(len({item.metadata["retained_normalized_caption"] for item in comparison.values()}), 1)
        for item in comparison.values():
            self.assertLessEqual(len(item.ids), 42)
            self.assertTrue(item.metadata["truncated"])
            self.assertEqual(item.metadata["occurrence_count"], 1)
            self.assertTrue(caption.startswith(item.metadata["retained_normalized_caption"]))
            self.assertTrue(item.metadata["serialized_text"].endswith("<end>\n<assistant>"))
        self.assertEqual(comparison["learned"].ids, comparison["init"].ids)

    def test_cannot_discard_earlier_content_to_save_late_marker(self):
        compiler = NativePromptCompiler(FixtureTokenizer(), "<s>", 4, 24)
        with self.assertRaisesRegex(ValueError, "Cannot fit caption"):
            compiler.compile("long ordinary scene description before [trigger]", require_trigger=True)
        report = compiler.report(["marker absent", "before [trigger] after"], ["[trigger] short"], "named")
        self.assertFalse(report["passed"])
        self.assertIn("missing", report["failures"][0]["error"])


if __name__ == "__main__":
    unittest.main()
