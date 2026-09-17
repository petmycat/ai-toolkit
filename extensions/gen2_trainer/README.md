# Gen2 trainer for Ideogram 4

## V2: standalone activator

The token-only v2 implementation uses `gen2.schema_version: "2.0.0"` with the
same `type: gen2_trainer` registration. Start with the
[v2 guide](docs/v2.md), [v2 smoke](config/smoke_gen2_ideogram4_v2.example.yaml),
and [500-update pilot](config/pilot_gen2_ideogram4_v2.example.yaml).
Its vectors occupy the original trigger positions; both Qwen and Ideogram stay
frozen. V2 has its own lifecycle, one optimizer, small token packages, and paired
base/named-phrase/initial/learned visual comparisons. Real CUDA and visual
acceptance remain VM checks. The sections below document **v1**.

## V1

An ai-toolkit extension for one target style, trained from captioned images. It
implements the approved v1 specification with four coupled parameter families:
diffusion LoRA, normalized learned suffix tokens, masked encoder LoRA, and
per-block time gates. The original diffusion model, text encoder, and VAE stay
frozen.

**Delivery status:** the implementation and local tests are ready for VM
validation. A real Ideogram / CUDA / `adamw8bit` acceptance pass is still required
on the user's RTX PRO 6000 96 GB VM. Local mathematical and persistence tests
do not establish that production backend acceptance or the style hypothesis.
See [VALIDATION.md](VALIDATION.md).

The original repository `gen2/` directory is read-only to this extension and to
the implementation task. Code, tests, examples, and the byte-preserved reference
live here. Runtime outputs use the configured training folder, normally
`output/<run-name>/gen2/`.

## Run on the VM

Use the VM's existing ai-toolkit environment and run these commands from the
repository root. Edit the YAML dataset path and, if needed, the model and encoder
paths to refer to weights already available on the VM. Captions describe image
content; a target-style name or description is not required. Existing native
image loading, resizing, buckets, and latent caches are reused.

### 1. Validate configuration without loading models

```bash
python -m extensions.gen2_trainer validate extensions/gen2_trainer/config/smoke_gen2_ideogram4.example.yaml
```

This checks the Gen2 schema, conflicts, phase totals, optimizer selection and
local scheduler horizons. It prints the resolved configuration. Add
`--output output/gen2-smoke-resolved.yaml` to save it. The ordinary native config
parser handles `${VARIABLE}` and `[name]` substitutions. Model paths, installed
GPU packages and actual native constructors are verified when the real run loads.

Check every training/validation caption and enabled sampling prompt with the
native tokenizer before starting a run:

```bash
python -m extensions.gen2_trainer check-captions CONFIG.yaml --output output/caption-token-report.json --local-files-only
```

This loads tokenizer files only. Omit `--local-files-only` to allow fetching
missing tokenizer files; it never loads model weights or creates latent caches.
The JSON report contains source paths, full and retained token lengths, hashes
of both token sequences, and every overflow or truncation. With `error`, an
oversized input returns failure; with `truncate`, reported truncations are
accepted. Invalid or empty token sequences still fail.
Normal training runs the same check automatically before loading model weights
and saves the report as `gen2/caption_token_report.json` under the run output.

`model.model_kwargs.max_text_length` counts the full chat-wrapped caption plus
the learned tokens. Its default remains 2,048; the user-approved maximum is now
3,072. Four learned tokens leave 3,068 original positions at the larger budget.
`gen2.conditioning.overflow_policy` supports two explicit choices:

- `error` (default): require the complete serialized caption and learned suffix
  to fit; report all oversized inputs and stop before model loading.
- `truncate`: retain the first `max_text_length - num_tokens` caption/chat tokens
  and discard only the excess token tail. Teacher, neutral student and styled
  student use exactly the same retained prefix; the styled path appends all
  learned positions. Source caption files and their full text are preserved.

Truncation follows the native tokenizer's right-side token-cut behavior, with
the additional Gen2 suffix reservation. It does not summarize or rewrite the
caption, split it into chunks, repair cut JSON, or restore removed chat endings.
Startup prints every truncated source with its full length, retained length and
discard count, and saves the complete report. Runtime conditioning metadata
records the full sequence length/hash as well as the actual encoded token IDs.
The user's retry3 config selects `truncate` with the 3,072-position total cap.

Longer inputs increase compute and memory use. The native model's ability to
accept variable lengths does not establish equal quality at every length.
Changing the token budget or overflow policy changes the strict resume contract;
use a new run name after either change.

### 2. Run the six-update smoke configuration

```bash
python run.py extensions/gen2_trainer/config/smoke_gen2_ideogram4.example.yaml
```

The update sequence is **D, D, A, D, A, G**:

- D updates diffusion LoRA only.
- A updates the learned tokens and masked encoder LoRA together, using two
  independent native optimizers.
- G updates time gates only.

This gives family counts D=3, E=2, T=2 and G=1. The smoke example keeps native
DiT quantization, both forms of gradient checkpointing, `adamw8bit`, rank-32
diffusion LoRA and all four learned tokens. It reduces image resolution to 256,
sampling to four steps and numerical probes to one example. Its images check
execution and comparisons; use the actual configuration for image-quality work.
Initial and boundary probes, spectra, and scheduled image ablations make it more
expensive than six optimizer steps alone.

If initialization fails before the initial checkpoint is published, use a new
`config.name` for the retry and retain the failed run's logs. Such a run cannot
resume from an unpublished checkpoint. A code-only checkpoint-context fix does
not require rebuilding image latents when the image/VAE settings are unchanged.
See [image-latent caching](docs/visual_diagnostics.md#what-image-latent-caching-changes)
for the cache's scope and validity limits.

### 3. Run strict continuation and package acceptance

```bash
python -m extensions.gen2_trainer acceptance extensions/gen2_trainer/config/smoke_gen2_ideogram4.example.yaml --output output/gen2_acceptance --split-after 3
```

Use a new or empty acceptance output directory. This command performs real
training in a continuous run and a separate interrupted/resumed run. It loads the
training backend three times in sequence and then performs an independent
inference-package load. It checks:

- Actual D/A/G parameter changes, inactive-family optimizer states and counters.
- Real A-step changes to the features consumed by the frozen DiT and its output.
- Native versus suffix-encoder prefix agreement at all 13 activation taps.
- Every saved master tensor, native optimizer/scheduler/scaler state, loader
  state, evaluation state and training RNG for continuous versus resumed runs.
- Live versus freshly loaded package predictions and generated image pixels.

`acceptance.json` reports the environment, checks and failures. The command also
exports diagnostic ZIPs. Floating-tensor comparisons are exact by default;
explicit `--atol` and `--rtol` values are recorded if supplied. Image pixels and
integer/counter/RNG state comparisons stay exact. Investigate a failure before
changing tolerances, checkpointing, precision or quantization.

The user runs these commands manually and supplies results under local `gen2/`.
The extension does not launch a remote VM or a long run automatically.

### 4. Run the actual experiment

```bash
python run.py extensions/gen2_trainer/config/train_gen2_ideogram4.example.yaml
```

The fully annotated actual example has 450 warmup D updates, 2,100 alternating
refinement updates in D,D,D,D,A cycles, and 450 gate-calibration updates. Its
optimizer-local horizons are D=2,130, E=420, T=420, G=450. Use a fresh run name
for a fresh experiment. It needs at least four real captioned training images for
the default fixed probes. Optional held-out images are supplied explicitly;
training probes remain labeled as training data.

## Checkpoint and resume

Complete packages are saved under:

```text
output/<run-name>/gen2/checkpoints/update_00000000/
output/<run-name>/gen2/checkpoints/update_00000006/
```

Each package contains all four fp32 master states, native inference exports,
optimizer/scheduler/scaler state, fixed probe/reference state, training and loader
RNG/cursor state, immutable specification bytes, metadata and checksums. A complete
marker is published only after the temporary directory is fully written. Zero,
phase-boundary and final checkpoints are protected from rolling retention.

To resume, set `gen2.checkpoint.resume_from` to a complete package directory and
run the ordinary launcher again. Keep the training mechanism and source identities
unchanged. Incompatible settings, missing components, changed weights/tokenizer,
modified specification bytes or damaged checksums abort before training continues.
Failed or partly committed updates require reload from the last complete package.

`num_workers: 0` is the reference loader mode. The loader restores ordering,
bucket indices, epoch and cursor, replays prior native transforms under saved RNG,
and restores the training RNG at the checkpoint boundary. Native dataset
initialization uses isolated `training_seed + expanded_dataset_index` seeds, so
cold-cache preparation does not consume the next training noise draw. Exact
continuation still requires the same data/cache bytes and environment. Worker
prefetch, cache regeneration with random transforms, changed hardware, quantizers
or CUDA versions cannot be treated as an established bitwise replay guarantee.

## Complete-package inference

### Independent image and checkpoint schedules

`sample.sample_every` is the sole repeating image interval, measured in committed
updates. `sample.sample_start_step` delays regular images; initialization is
controlled separately by `train.skip_first_sample`. `train.disable_sampling`
disables both. Saves, stage boundaries, and the final update never force images.
For example, 12 updates with `sample_every: 6` and initialization enabled produce
images at 0, 6, and 12. With `sample_every: 5`, images occur at 0, 5, and 10.

During sampling the console prints the update, the number of new images after
deduplication, and each image's position, prompt ID, seed and mode. It reports
the first denoising step, approximately each quarter, and the last step, then
image completion with elapsed time. Failures name the active image and propagate
normally. These progress messages do not add model evaluations or device
synchronization.

`gen2.evaluation.milestone_every` only expands a sample that is already due and
whose update is divisible by that interval: it adds milestone modes and extra
seeds. It does not create another sampling clock. Ordinary samples use preview
modes and the primary seed. Both lists can contain the same six or seven visual
comparisons. Initialization is an expanded milestone because its update is zero.

`save.save_every` controls regular checkpoint saves. Gen2 separately retains
initialization, phase-boundary, and final checkpoints. Thus the reviewed 12-update
smoke with boundaries 0/2/10/12 and `save_every: 6` has five checkpoint directories
(0, 2, 6, 10, 12), while `sample_every: 6` generates only three image sets (0, 6, 12).

### Load a complete package

```bash
python -m extensions.gen2_trainer infer output/gen2_style_smoke/gen2/checkpoints/update_00000006 --prompt "A cat beside a vase <gen2style>" --output output/gen2_inference.png --width 256 --height 256 --steps 4
```

The loader verifies frozen model/tokenizer identities and restores all four
components. It uses the same generation function as training previews. A PNG and
JSON metadata are saved. Use a new output filename for each result.

With no `--mode`, a literal trigger selects learned conditioning. Missing triggers
follow `gen2.inference.missing_trigger_policy`: `learned_neutral` keeps diffusion
LoRA with neutral conditioning; `base_bypass` bypasses personalization. Trigger
matching is literal and case-sensitive; `[trigger]` expands first, all literal
occurrences are removed, and only outer whitespace is trimmed.

Explicit diagnostic modes override trigger routing:

- `full`: current suffix, encoder adapter, diffusion LoRA and current gates.
- `neutral_lora_on`: neutral conditioning with current diffusion LoRA/gates.
- `base`: neutral conditioning with personalization bypassed.
- `base_with_tokens`: current suffix, encoder adapter off, diffusion LoRA off.
- `base_with_conditioning`: learned conditioning with diffusion LoRA bypassed.
- `conditioning_init`: saved initial suffix, encoder adapter off.
- `encoder_adapter_off`: current suffix, encoder adapter off.
- `tokens_init`: saved initial suffix with the current encoder adapter.
- `gates_one`: full components with all gates equal to one.
- `gates_time_mean`: full components with each gate's fixed-grid mean.
- `full_uncond_half`: full conditional path, with the same diffusion LoRA also
  applied to the unconditional pass at absolute strength 0.5.
- `full_uncond_full`: full conditional path, with the same diffusion LoRA also
  applied to the unconditional pass at absolute strength 1.0.

The CFG unconditional pass is image-only. Production routing and the original
diagnostic modes disable personalization there. The two explicit `full_uncond_*`
experiments require guidance greater than one and enable only the trained diffusion
LoRA on that pass, with the same time gates as the conditional pass. Their strengths
are independent of `--strength`, which controls the conditional diffusion residual.
Learned tokens and the encoder adapter remain conditional-only. An optional existing
native unconditional adapter remains frozen, is kept fixed across modes, and has its
complete weight identity checked. `--strength 0` leaves triggered learned conditioning
present; it also leaves the explicitly requested `full_uncond_*` strength in effect.
Use `--mode base` for a base comparison.

### Original unconditional transformer

To run CFG with Ideogram's separate original unconditional weights, set:

```yaml
model:
  unconditional_lora_path: null
gen2:
  inference:
    unconditional_model_path: "ideogram-ai/ideogram-4-fp8"
```

The new path is a model repository ID or local repository root containing
`unconditional_transformer/`. It uses the native strict loader and the same
transformer dtype, quantization and offload settings as the conditional model.
The original unconditional weights stay frozen and run only during CFG sampling.
All training students and the neutral teacher still use the conditional model.

For `full`, `full_uncond_half` and `full_uncond_full`, the image-only model uses
the same live trained diffusion LoRA factors at strengths 0, 0.5 and 1.0. Each
residual is evaluated on that model's own activations, with the current shared
time gates. There is no second trainable LoRA or optimizer. Checkpoint reloads
restore the one diffusion parameter set used by both models.

The original model cannot be combined with a native unconditional correction
LoRA. The default `unconditional_model_path: null` preserves existing behavior:
the image-only pass reuses the conditional backbone, optionally with the frozen
native correction adapter. Samples record which backend and source were used;
checkpoints verify the complete frozen original model identity when enabled.
The original weights load again from their source when restoring a package;
they are not duplicated inside each checkpoint.

The user's retry2 config enables this original model. Local CPU tests cover
branch routing, shared factors and strict reload identity; the full quantized
model still needs the user's VM smoke run. More VRAM is required for the second
backbone, and its component may need downloading if it is not cached on the VM.

See the [visual diagnostics and plain-language parameter guide](docs/visual_diagnostics.md)
for the six matched comparisons, an embedding-off reference, and configuration advice.
These extra experiments are opt-in; existing default mode lists remain unchanged.

**A bare diffusion LoRA file is incomplete.** The package includes tokens,
encoder adapters and gates. Native PEFT export also omits alpha tensors; the
package master state and target manifest preserve the configured alpha/rank.
Use the complete loader as the authoritative inference path.

## Diagnostics and result sharing

```bash
python -m extensions.gen2_trainer inspect output/gen2_style_smoke/gen2/checkpoints/update_00000006
python -m extensions.gen2_trainer summarize output/gen2_style_smoke/gen2
python -m extensions.gen2_trainer export output/gen2_style_smoke/gen2 --output output/gen2_smoke_diagnostics.zip
```

Add `--include-images` to export individual images and contact sheets. Export
reads the source without changing it, and excludes model weights and original
dataset images. Summarize writes derived summaries into the run directory, so
use it on writable output directories. Export can read results placed under the
protected local `gen2/` folder.

Core records include per-example states/losses, per-update ownership and native
optimizer rates, per-projection token-region/time-bin residuals, fixed 2x2
conditioning/LoRA interventions, isolated loss-gradient comparisons, token
geometry, gate curves and thin-QR LoRA spectra. Coordinate selections are stored
once. Core recording has bounded queues, backpressure, rotation and explicit hard
budget errors. Optional tensor dumps have separate packet/byte limits. Sampling
and diagnostics preserve training RNG. Host timers are labeled approximate for
asynchronous CUDA work.

`human_ratings.csv` starts with empty score fields. Add your own style fidelity,
content fulfillment, artifact and suspicious-similarity ratings. The summary
keeps D/A/G objectives separate, distinguishes training/validation data, and
reports matched comparisons only where actual ratings exist. Numeric interaction
or reconstruction metrics are not automated style scores.

## Initial optimizer and backend boundaries

All optimizer instances come from the existing native factory. Initial accepted
names are `adam`, `adamw`, `adagrad`, `lion`, `prodigy`, `adafactor`, bare
`dadaptation`, `adam8bit`, **`adamw8bit`**, `lion8bit` and `ademamix8bit`.
Optional packages must exist in the VM environment. This selection is an
implementation boundary, not a claim that every optional optimizer has already
passed real-model validation. Adam, AdamW and Adagrad have local native-factory
continuation tests. `adamw8bit` is the production acceptance priority.

Custom/experimental variants, Automagic2, custom `adam8`/`adamw8` and broken named
DAdapt factory routes are deferred as approved. No optimizer algorithm was
repaired or duplicated. Per-family scalar nulls inherit `train.*`; kwargs maps
merge shallowly over copies of native maps. Scheduler clocks count that family's
successful updates. A family with zero scheduled updates has no constructed
scheduler and an explicit inactive descriptor in its saved state.

V1 uses one CUDA process/GPU and the native Ideogram 4 / Qwen backend. It rejects
extra style adapters, cached learned text features, parameter swapping, EMA,
dropout, compilation and incompatible loss/noise/caption changes. Optional native
quantization/offload settings still need actual A-gradient acceptance for the
selected profile. bf16 is the example compute precision; all trainable masters
are fp32. fp16 requires an accelerator GradScaler, configured through the native
accelerator launch environment. No fallback silently changes the chosen profile.

## Development checks and integration inventory

```bash
python -B -m pytest extensions/gen2_trainer/tests -q -p no:cacheprovider
```

The mathematical fixtures and source-isolated native helper tests deliberately
avoid loading production model weights. Real-backend acceptance is the separate
VM command above. [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) records the
approved scope and reviewed native symbols. Existing-file changes are limited to
the Git ignore exception and an optional dataset-class hook in the native loader
factory. Stock callers retain the same default dataset class and behavior.
