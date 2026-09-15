"""Contact sheets preserve matched comparisons even after sample deduplication."""
from PIL import Image, ImageDraw, ImageFont
import pytest

from extensions.gen2_trainer.evaluation import contact_sheet_label, make_contact_sheet


MODES = ["base_with_tokens", "base_with_conditioning", "encoder_adapter_off", "full",
         "full_uncond_half", "full_uncond_full", "base"]


def item(mode, seed, color, prompt_id="p000"):
    return Image.new("RGB", (20, 20), color), {
        "prompt_id": prompt_id, "seed": seed, "ablation_mode": mode}


def test_matched_seeds_remain_rows_and_modes_remain_columns():
    # Simulate partial deduplication: the first mode at seed 42 and the fourth
    # mode at seed 43 already exist and are absent from this generation batch.
    images = [item(MODES[1], 42, "red"), item(MODES[5], 42, "blue"),
              item(MODES[0], 43, "green"), item(MODES[6], 43, "yellow")]
    sheet = make_contact_sheet(images, MODES)
    assert sheet.size == (7*256, 2*318)
    assert sheet.getpixel((256 + 128, 128)) == (255, 0, 0)
    assert sheet.getpixel((5*256 + 128, 128)) == (0, 0, 255)
    assert sheet.getpixel((128, 318 + 128)) == (0, 128, 0)
    assert sheet.getpixel((6*256 + 128, 318 + 128)) == (255, 255, 0)
    assert sheet.getpixel((1, 1)) == (238, 238, 238)
    assert sheet.getpixel((3*256 + 1, 318 + 1)) == (238, 238, 238)
    assert images[0][0].size == (20, 20)


def test_distinct_prompts_with_same_seed_get_distinct_rows():
    sheet = make_contact_sheet([item("full", 42, "red"), item("full", 42, "blue", "p001")], ["full"])
    assert sheet.size == (256, 636)
    assert sheet.getpixel((128, 128)) == (255, 0, 0)
    assert sheet.getpixel((128, 318 + 128)) == (0, 0, 255)


def test_labels_explain_all_six_controls_and_embedding_off_reference():
    metadata = {"prompt_id": "p000", "seed": 42}
    labels = [contact_sheet_label(mode, metadata) for mode in MODES]
    assert [lines[1] for lines in labels] == [
        "D off | TE off | U on", "D off | TE on | U on", "D on(1) | TE off | U on",
        "D on(1) | TE on | U on", "D on(1) | TE on | U on", "D on(1) | TE on | U on",
        "D off | TE off | U off"]
    assert [lines[2] for lines in labels] == ["Uncond D: 0"]*4 + ["Uncond D: 0.5", "Uncond D: 1", "Uncond D: 0"]
    draw, font = ImageDraw.Draw(Image.new("RGB", (256, 256))), ImageFont.load_default()
    assert all(draw.textlength(line, font=font) <= 248 for lines in labels for line in lines)


def test_labels_use_recorded_strengths_and_actual_cfg_execution():
    metadata = {"prompt_id": "p123", "seed": 123, "conditional_lora_strength": .75,
                "unconditional_lora_strength": .5, "route": {"lora_enabled": True, "styled": True,
                    "adapter_enabled": False, "token_mode": "init"}}
    lines = contact_sheet_label("full", metadata)
    assert lines[1] == "D on(0.75) | TE off | U init"
    assert lines[2] == "Uncond D: 0.5"
    assert lines[3] == "p123 | seed 123"
    metadata["unconditional_branch_executed"] = False
    assert contact_sheet_label("full", metadata)[2] == "Uncond D: skipped"


@pytest.mark.parametrize("mode,tokens,adapter", [
    ("conditioning_init", "init", "off"), ("tokens_init", "init", "on"),
    ("neutral_lora_on", "off", "off"), ("gates_one", "on", "on"),
    ("gates_time_mean", "on", "on")])
def test_original_ablation_modes_remain_labeled(mode, tokens, adapter):
    lines = contact_sheet_label(mode, {"prompt_id": "p000", "seed": 42})
    assert f"TE {adapter} | U {tokens}" in lines[1]


def test_duplicate_or_unrequested_images_are_rejected():
    first = item("full", 42, "red")
    with pytest.raises(ValueError, match="Duplicate"):
        make_contact_sheet([first, first], ["full"])
    with pytest.raises(ValueError, match="unrequested"):
        make_contact_sheet([first], ["base"])
