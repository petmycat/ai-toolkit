"""Fixed references, independently scheduled images, and compact comparisons."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

from ..diagnostics import isolated_rng, module_hash
from ..recording import write_json, sha256, append_rating_template
from .inference import generate


LABELS = {"base": "Base / marker removed", "named": "Base / named phrase",
          "init": "Base / initial random activator", "learned": "Base / learned activator"}


def make_contact_sheet(rows, modes, update):
    from PIL import Image, ImageDraw
    width, height, header = 512, 384, 60
    sheet = Image.new("RGB", (width * len(modes), (height + header) * len(rows)), "#202020")
    draw = ImageDraw.Draw(sheet)
    for row_index, (label, paths) in enumerate(rows):
        for col, mode in enumerate(modes):
            x, y = col * width, row_index * (height + header)
            title = LABELS[mode]
            if mode == "learned" and update == 0:
                title = "Initial = current (0 updates)"
            draw.text((x + 8, y + 7), title, fill="white")
            draw.text((x + 8, y + 28), f"{label} | update {update}", fill="white")
            with Image.open(paths[mode]) as source:
                image = source.convert("RGB")
                image.thumbnail((width, height))
                sheet.paste(image, (x + (width-image.width)//2, y + header + (height-image.height)//2))
    return sheet


class V2Evaluation:
    def __init__(self, backend, config, recorder, root):
        self.backend, self.config, self.recorder, self.root = backend, config, recorder, Path(root)
        self.references = {}

    def state_dict(self):
        return {"references": dict(self.references)}

    def load_state_dict(self, state):
        self.references = dict(state["references"])

    def _reference(self, key):
        record = self.references.get(key)
        if record is None:
            return None
        path = (self.root / record["path"]).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("Invalid reference-image path in resumed evaluation state")
        if not path.is_file():
            # A resume into a new output directory recreates these fixed
            # references from the saved initialization with isolated RNG.
            return None
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Fixed visual reference changed: {path}")
        return path

    def sample(self, update):
        from toolkit.config_modules import GenerateImageConfig
        options, evaluation = self.config["sample"], self.config["gen2"]["evaluation"]
        modes = ["base"] + (["named"] if evaluation["named_phrase"] else []) + ["init", "learned"]
        comparisons, requests = [], []
        for prompt_index, prompt in enumerate(options["prompts"]):
            compiled = self.backend.compiler.comparison(prompt, named_phrase=evaluation["named_phrase"])
            for seed in evaluation["seeds"]:
                paths = {}
                comparisons.append((f"prompt {prompt_index} | seed {seed}", paths))
                for mode in modes:
                    key = f"p{prompt_index:03d}_s{seed}_{mode}"
                    reference = self._reference(key) if mode != "learned" else None
                    if reference is not None:
                        paths[mode] = reference
                        continue
                    if update == 0 and mode == "learned":
                        # Same bank; alias to init after that image is available.
                        continue
                    requests.append((prompt_index, prompt, seed, mode, key, compiled[mode], paths))
        print(f"[Gen2 v2 sampling] update {update}: {len(requests)} new images; fixed references reused", flush=True)
        token_hash = module_hash(self.backend.tokens)
        ratings = []
        with isolated_rng(self.config["gen2"]["execution"]["training_seed"]):
            for index, (prompt_index, prompt, seed, mode, key, compiled, paths) in enumerate(requests, 1):
                started = time.perf_counter()
                prefix = f"[Gen2 v2 sampling] update {update} image {index}/{len(requests)} | seed {seed} | {mode}"
                print(prefix + " | starting", flush=True)

                def progress(completed, total):
                    if completed in {1, total, *((total*q+3)//4 for q in (1, 2, 3))}:
                        print(f"{prefix} | denoising {completed}/{total} | {time.perf_counter()-started:.1f}s", flush=True)

                image, metadata = generate(self.backend, prompt, mode, seed=seed, compiled=compiled,
                    width=options["width"], height=options["height"], steps=options["sample_steps"],
                    guidance=options["guidance_scale"], progress_callback=progress)
                folder = "references" if mode != "learned" else "samples"
                filename = key + (f"_u{update:08d}" if mode == "learned" else "") + ".png"
                destination = self.root / folder / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                native = GenerateImageConfig(prompt=prompt, output_path=str(destination), output_ext="png",
                    seed=seed, width=options["width"], height=options["height"],
                    num_inference_steps=options["sample_steps"], guidance_scale=options["guidance_scale"],
                    add_prompt_file=False)
                native.save_image_atomic(image)
                relative = destination.relative_to(self.root).as_posix()
                image_hash = sha256(destination)
                metadata.update(path=relative, logical_update=update, token_state_sha256=token_hash,
                    prompt_id=f"p{prompt_index:03d}", ablation_mode=mode, checkpoint_hash=token_hash,
                    image_id=hashlib.sha256(f"{key}:{update}:{image_hash}".encode()).hexdigest(),
                    image_sha256=image_hash, fixed_reference=mode != "learned")
                write_json(destination.with_suffix(".json"), metadata)
                self.recorder.record("samples/manifest", metadata)
                ratings.append(metadata)
                paths[mode] = destination
                if mode != "learned":
                    self.references[key] = {"path": relative, "sha256": image_hash}
                print(f"{prefix} | complete in {time.perf_counter()-started:.1f}s", flush=True)
        if update == 0:
            for _, paths in comparisons:
                paths["learned"] = paths["init"]
            self.recorder.event("initial_current_images_identical", alias="learned -> init", update=0)
        append_rating_template(self.root / "human_ratings.csv", ratings)
        if evaluation["make_contact_sheets"]:
            path = self.root / "samples" / f"u{update:08d}_comparison.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            make_contact_sheet(comparisons, modes, update).save(path)
        self.recorder.event("visual_sampling_complete", images=len(requests), visual_acceptance="pending_user_review")
        print(f"[Gen2 v2 sampling] update {update}: complete", flush=True)
