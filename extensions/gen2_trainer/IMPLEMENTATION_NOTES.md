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
5. The user subsequently required visual sampling to follow its own interval.
   `sample.sample_every` now exclusively controls repeating image events, with
   optional initialization controlled by `skip_first_sample`. Checkpoint saves,
   stage boundaries and final updates do not force images. `milestone_every`
   only expands modes/seeds at an already-due sample. This explicitly supersedes
   the reference's mandatory boundary/milestone image scheduling. Numerical
   boundary probes and protected checkpoint saves retain their original rules.
   The user's 12-update smoke uses `sample_every: 6`, producing images at 0/6/12.
6. On 2026-09-16 the user approved a 3,072-position total text budget after a
   complete 2,055-token training caption plus four learned positions exceeded
   the original 2,048 limit. The configurable maximum is now 3,072, matching
   this checkout's native Ideogram default. The default remains 2,048 for
   existing configs. This supersedes only the reference's upper bound:
   `overflow_policy: error`, complete chat serialization, identical neutral and
   styled prefixes, and the M-position reservation remain enforced. The
   immutable specification remains unchanged. Longer inputs need more compute
   and memory; the new budget still requires real VM acceptance.

Caption preflight uses the same digest, chat template and non-truncating
tokenization as the encoder. It checks every source caption, held-out caption
and enabled sampling prompt before model weights or latent caches load, reports
all offending paths together, and saves `caption_token_report.json` in the run
output. The `check-captions` CLI exposes this check with tokenizer files only.

### Original unconditional model: user-approved extension on 2026-09-16

The user authorized the original separate unconditional backend before retry2,
to test community reports that applying a conditional-trained LoRA to both
models improves generation. `gen2.inference.unconditional_model_path` opts into
the original `unconditional_transformer` component. A null default preserves
existing behavior and legacy resume contracts. The private retry2 YAML selects
`ideogram-ai/ideogram-4-fp8`; native `model.unconditional_lora_path` remains null.
Config validation rejects selecting both the original model and its native
correction-LoRA approximation.

The new transformer is frozen and inference-only. D/A/G objectives, captioned
teacher/student routing, all four trainable families and optimizer horizons
remain unchanged. All five adapted projections per block on the original model
read the exact existing diffusion LoRA parameter objects, with the same residual
equation and gates. The factors operate on each backbone's own activations.
There is no copy, merge or synchronization operation. Bindings are fully validated
before attachment, and shared target manifest entries count zero new parameters.
The image-only CFG branch still receives zero text positions and requires
`torch.no_grad`; the existing 0/0.5/1.0 diagnostic modes select its style strength.

Loading uses native strict state loading, FP8 scale reconstruction, dtype,
quantization and offload policy. RNG isolation prevents this additional frozen
model construction from changing the training adapters' initialization. The
complete original model state, including quantizer buffers, joins frozen-state
and package identity checks. Both models are loaded before initial hashes and
adapter binding; package reloads verify identity before restoring any learned
parameters. Samples record backend kind, source and shared parameter family.
No original model weights are embedded into the personalization checkpoints.

The immutable v1 reference remains unchanged. This approved extension changes
inference backend selection; it does not assert that unconditional training is
needed or that one diagnostic strength is better before reviewing VM results.

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
their existing behavior. Gen2 also rejects native cache-time removals and checks
membership independently for each source and expanded resolution; a surviving
copy at another resolution cannot hide a missing example. No dataset algorithm
is forked.

The native Ideogram `_load_transformer` accepts an optional component name,
defaulting to the existing `transformer`. Gen2 selects `unconditional_transformer`
through that seam; strict loading, FP8 handling and rotary setup stay native.
Stock calls keep their original behavior.

The native Ideogram transformer has one additional optional checkpoint callback
(`_gradient_checkpointing_func`), falling back to the original torch function
for stock callers. Gen2 installs a callback that captures each block forward's
branch, gate tensor, and adapter flags and rebinds them during backward replay.
Keeping the outer branch scope open is insufficient when CUDA autograd performs
recomputation in another Python context. Replays preserve the original gate
autograd graph, suppress duplicate diagnostics, and restore ambient state even
on exceptions. Both diffusion and encoder replay contexts support repeated
entries for chunked `autograd.grad` probes. The transformer forward, attention,
checkpoint boundaries and gradients remain native. All pipeline logic and tests
live in this extension; no optimizer repair or broad trainer refactor.

## Validation status

Implementation validation results and VM acceptance status are recorded in
`VALIDATION.md`. Scientific usefulness requires the user's actual image results;
implementation checks do not establish the style-learning hypothesis.
