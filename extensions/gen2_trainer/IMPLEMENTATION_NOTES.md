# Gen2 implementation decisions and integration inventory

## Authority

The byte-preserved `docs/specs/gen2_trainer_v1.md` is the original reference.
SHA-256: `6880f48649280c02fe7e73bf9b3b244eef4da6ca63da02226bc876cae5df03b8`.
The original local `F:/ai-toolkit/gen2/` directory is protected from writes.

The user approved these decisions in this task before authorizing implementation:

1. A parameter family with zero total scheduled updates has a saved inactive
   scheduler descriptor with horizon zero; no scheduler object is constructed.
2. Initial compatibility targets standard native optimizers, especially
   `adamw8bit`. Custom/experimental optimizers are deferred. Automagic2, custom
   `adam8`/`adamw8`, and broken named D-Adaptation routes are excluded. This
   explicitly supersedes the reference's requirement that every native optimizer
   must initially be available. No optimizer algorithm is duplicated or repaired.
3. The user manually runs smoke/actual configurations on an RTX PRO 6000 96 GB
   VM, using its existing dependencies and datasets, and supplies results under
   the local `gen2/` directory. A local test pass does not establish VM acceptance.
4. On 2026-09-15 the user requested six matched visual comparisons, including
   the same trained diffusion LoRA on the image-only unconditional pass at
   strengths 0, 0.5 and 1.0. The explicit `full_uncond_half` and `full_uncond_full`
   inference diagnostics supersede the reference's unconditional-personalization
   prohibition for those two experiments only. They require no-grad execution
   and CFG greater than one. Conditional tokens/encoder adapters stay absent
   from the unconditional pass; training and default production routing retain
   the original behavior. Both passes use the same time-gate profile. The
   immutable specification remains unchanged. `base_with_tokens` supplies the
   requested tokens-only reference.

## Inventory recorded before implementation

Reviewed checkout: `30c82a877455e152e8ab8af6a21e6d4aa2e35fa8`. The inspected
trainer/Ideogram/factory integration files match reference revision
`f56b5a1d405f819c74724228564e99982624c186`.

- Register `toolkit.extension.Extension` with lazy `get_process`.
- Reuse native process scaffolding and configuration types. Gen2 owns the run
  loop, phase masks, four optimizer/scheduler instances, accumulation and commits.
  The stock `BaseSDTrainProcess.run`/`SDTrainer.hook_train_loop` are not called.
  The implemented bridge composes `BaseTrainProcess`, preserving native run
  naming, config saving and logger interfaces without inheriting its SD loop.
- Reuse `Ideogram4Model`, native loader/quantization/placement, `encode_images`,
  `decode_latents`, `get_train_scheduler`, `get_bucket_divisibility`.
- Reuse `get_dataloader_from_datasets`, DTOs, native image transforms and latent
  caches. Derive canonical prompts from raw captions, bypassing native comma
  rejoining and trigger reinsertion. Native dataset writes require writable data
  roots; source data under the protected local `gen2` root is rejected.
- Reuse `digest_caption_string`, Qwen chat template, loaded embedding/decoder/
  rotary modules and native causal mask. Add a differentiable suffix encoder
  sibling inside this extension; preserve channel/tap interleave.
- Reuse `LoRASpecialNetwork`/`LoRAModule` and native serialization; extend only
  residual masking/gating. Keep native network multiplier scalar; differentiable
  per-example gates are applied to residuals within scoped contexts.
  Native transformer/PEFT construction replaces scalar alpha with rank; the
  existing `modules_dim`/`modules_alpha` seam preserves the requested per-target
  scaling. Complete master states and module manifests retain alpha on reload.
- Reuse `pad_text_features`, `predict_velocity`, native time table and
  `add_noise`, `get_ideogram4_sigmas`, VAE and image-save machinery. Explicit CFG
  contexts disable personalization on the image-only unconditional pass except
  for the two explicitly requested inference experiments described above.
- Reuse native `get_optimizer`, `get_lr_scheduler`, accelerator backward and
  precision helpers. Resume restores all parameter families together.
- Gen2 supplies coherent atomic checkpoints, protected retention, bounded JSONL
  recording, fixed probes, gradient diagnostics, matched sampling and export.
  A small labeled PIL contact-sheet composer is local to evaluation; the source
  inventory found no general native image-grid utility. Individual sample saves
  use native `GenerateImageConfig.save_image_atomic`.

## Existing-file changes

`.gitignore` has a narrow exception for the new extension. The native data-loader
factory gains an optional `dataset_class` injection argument, defaulting to the
same `AiToolkitDataset` as before. This lets Gen2 inherit native preprocessing
while overriding only the native retry/replacement policy: a failed source image
must abort rather than silently substitute another sample. Stock callers retain
their existing behavior. No dataset algorithm is forked. All other implementation
and tests live in this extension; no optimizer repair or broad trainer refactor.

## Validation status

Implementation validation results and VM acceptance status are recorded in
`VALIDATION.md`. Scientific usefulness requires the user's actual image results;
implementation checks do not establish the style-learning hypothesis.
