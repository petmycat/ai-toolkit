# Visual diagnostics and the Gen2 settings

Gen2 learns three cooperating changes: a diffusion LoRA that changes how the image
is drawn, learned text vectors that carry the style signal, and a text-encoder
adapter that adjusts how those extra vectors are interpreted. A final calibration
stage learns when, during denoising, each part of the diffusion LoRA should have
more or less influence.

This guide explains the single-prompt visual comparison and the configuration
groups that control it. The [configuration schema](../config.py),
[inference routes](../inference.py), and [annotated actual example](../config/train_gen2_ideogram4.example.yaml)
are the implementation references. See the [README](../README.md) for launch and
package-loading commands.

## The six requested images, plus a base reference

Every comparison uses the same prompt, seed, initial noise, checkpoint, image
dimensions, sampling steps, and guidance setting. Only the named switches change.
"Embedding on" means the current learned suffix vectors are included. It does not
mean that the ordinary prompt's vocabulary embeddings are removed in other modes.

1. **`base_with_tokens`** — diffusion LoRA off, text-encoder adapter off, learned
   embedding on. This shows what the learned vectors can do through the frozen
   encoder and frozen diffusion model.
2. **`base_with_conditioning`** — diffusion LoRA off, text-encoder adapter on,
   learned embedding on. Compare with image 1 to see the encoder adapter's effect
   while the diffusion model stays unchanged.
3. **`encoder_adapter_off`** — diffusion LoRA on, text-encoder adapter off,
   learned embedding on. Compare with image 1 to see the diffusion LoRA's effect
   with the same simpler conditioning path.
4. **`full`** — all three learned components on for the conditional pass;
   diffusion LoRA strength **0** on the unconditional pass. Compare with image 3
   to see the encoder adapter's contribution when the diffusion LoRA is active.
5. **`full_uncond_half`** — the same conditional settings as image 4; diffusion
   LoRA strength **0.5** on the unconditional pass.
6. **`full_uncond_full`** — the same conditional settings as image 4; diffusion
   LoRA strength **1.0** on the unconditional pass.

The optional seventh image, **`base`**, turns off the learned suffix, encoder
adapter, and diffusion LoRA. The normal content prompt still goes through the
original model. Comparing this with image 1 gives a separate reference for the
learned embedding. Without this seventh image, all six requested images contain
the learned embedding, so they cannot directly show its effect relative to the
original model.

Use this order in both evaluation mode lists when every preview should contain
the same comparison:

```yaml
preview_modes:
  - base_with_tokens
  - base_with_conditioning
  - encoder_adapter_off
  - full
  - full_uncond_half
  - full_uncond_full
  - base
milestone_modes:
  - base_with_tokens
  - base_with_conditioning
  - encoder_adapter_off
  - full
  - full_uncond_half
  - full_uncond_full
  - base
```

These are explicit diagnostic routes. They select their switches regardless of
whether a literal trigger is present. Ordinary inference without an explicit mode
continues to use trigger-based routing.

### What the unconditional comparison changes

Ideogram's unconditional pass receives the current noisy image with **no text
tokens**. It has neither the learned suffix nor the encoder adapter. Images 5 and
6 apply the **same trained diffusion LoRA** to that image-only pass. They do not
train a second LoRA or copy the positive prompt into the unconditional pass.

The values 0.5 and 1.0 are **absolute LoRA strengths**, independent of the
conditional `gen2.inference.lora_strength`. With conditional strength 1.0, they
also happen to be half and equal strengths relative to the conditional setting.
If conditional strength is changed to 0.8, the two diagnostic strengths remain
0.5 and 1.0.

Both passes use the same current time gates when their LoRA is enabled. For
example, if a block's gate is 1.2 at a particular noise level, the unconditional
half-strength route scales that block's ordinary LoRA residual by `0.5 × 1.2`.
During warmup and refinement the gates are held at one; calibration can change
their curves. Disabling the LoRA removes its residual completely.

`model.unconditional_lora_path` is a separate native option for an already
existing, frozen unconditional adapter. The reviewed smoke sets it to `null`.
Keep it fixed across this experiment if using it in another run; it is not the
new 0/0.5/1.0 control.

### Why stronger unconditional LoRA need not mean stronger visible style

Classifier-free guidance (CFG) combines two denoising predictions: one with the
prompt, called `C`, and the image-only prediction, called `U`. At guidance 3:

```text
guided prediction = U + 3 × (C − U) = 3C − 2U
```

Changing the unconditional LoRA changes `U`, which enters with a negative
coefficient here. The result can alter style, content, contrast, or stability; it
does not have to increase style monotonically as the strength rises. Each update
also changes the image presented to the next denoising step. The formula explains
the comparison at one common noisy image, not a linear mixture of the final PNGs.

Native sampling executes CFG only when `guidance_scale > 1`. The reviewed value
3.0 therefore exercises the unconditional pass. At values at or below one, the
native sampler uses the conditional prediction directly, so these unconditional
strength choices cannot produce a useful comparison.

Compare images 4, 5, and 6 within the **same seed and checkpoint**. A preference at
one seed is useful evidence for that case. Repeating with the second fixed seed
helps distinguish a consistent preference from an accidental composition. This
single prompt can answer how the choices behave for that prompt; broader claims
require further prompts or datasets later.

## What the trigger and initializer actually do

`trigger_word` is a reserved switch. The compiler first replaces every
`[trigger]` placeholder with that literal string. It then removes every occurrence
of the reserved string from the content text. Matching is case-sensitive. The
model receives the remaining content prompt, followed by one learned suffix.

Three placeholders therefore do **not** create three sets of learned vectors or
put vectors at those positions. With `num_tokens: 4`, four learned positions are
appended once, after the complete original chat prompt. This preserves the
original prompt positions and gives the style its own extra positions.

For a structured JSON prompt, surrounding text remains after trigger removal.
A value such as `"[trigger] style medium"` becomes a generic style-medium phrase;
the native Ideogram caption helper also performs its normal JSON normalization.
Lighting descriptions, palette colors, object descriptions, and composition
remain explicit instructions in every image. They can constrain the appearance
even in the base reference. The reviewed single prompt is preserved for a
consistent comparison.

**An arbitrary string is a valid initializer choice.** `initializer_text` and
`trigger_word` have separate jobs even when they contain the same string. The
initializer is split into existing vocabulary tokens, and their existing
embedding vectors are averaged to create the starting direction. Small seeded
perturbations distinguish the learned positions. No new vocabulary entry is
registered, and the initializer is not required to name the target style.

This initialization is not pure random noise: an arbitrary character string is
still decomposed into vocabulary pieces that already have vectors. That is the
configured behavior, and the arbitrary-string preference can be kept. The
initializer seed controls the perturbations; it does not change how the string
is tokenized.

The reviewed prompt was checked with the already cached Qwen tokenizer, using the
native caption normalization and chat template: **1,269 original positions plus
4 learned positions = 1,273 of the 2,048-position limit**, leaving 775 positions.
This check used tokenizer revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` without downloads. Dataset captions have
their own lengths and are checked separately when encoded. Overlength input
raises an error rather than being silently cut short.

## Understanding the reviewed 12-update smoke

An update means one committed training decision. D changes the diffusion LoRA;
A changes both the learned embedding and the text adapter; G changes only the
time gates. Parameters outside the active group stay frozen for that update.

With warmup 2, refinement 8, calibration 2, and a refinement cycle of four D
updates followed by one A update, the sequence is:

```text
Update:  1  2 | 3  4  5  6  7  8  9 10 | 11 12
Kind:    D  D | D  D  D  D  A  D  D  D |  G  G
```

The optimizer clocks therefore advance **D: 9, embedding: 1, text adapter: 1,
gates: 2**. A partially completed refinement cycle stops at the specified stage
length; the scheduler does not add another A update to finish it.

This exercises all update types. One A update gives little opportunity for
visible conditioning learning, however. The encoder LoRA starts with one factor
at zero, so its other factor initially has zero loss gradient. The first A update
can change the zero-initialized factor and the learned embedding; a later A update
can exercise gradients through both LoRA factors. Weight decay is separate from
this statement about the loss gradient.

For a separate smoke intended to exercise more conditioning updates, a **1 D :
1 A** refinement cycle would give D6/embedding4/text-adapter4/G2 with the same
12-update stage lengths. That is an optional experiment, not a change required
for the existing 4:1 configuration. Neither short schedule establishes style
quality or convergence.

### Image sampling has its own interval

**`sample.sample_every` is the only repeating image-generation interval.**
Checkpoint saves, phase boundaries, and the final update do not independently
trigger images. `train.skip_first_sample: false` adds an initialization comparison
at update 0; setting it true skips that comparison. `train.disable_sampling: true`
disables all images while numerical diagnostics and checkpointing continue.

The selected smoke interval is **every 6 updates**, so its images occur at
**updates 0, 6, and 12**. The matching `evaluation.milestone_every: 6` expands
those already scheduled image events to use milestone modes and additional
seeds. It does not create another sampling schedule. With seven modes and seeds
42 and 43, the completed smoke produces **14 individual images at each event:
42 images total, plus three contact sheets**. The six requested modes alone
would produce 36 images. `additional_seeds: []` keeps the same single prompt and
halves the image count for this configuration.

When the two intervals differ, ordinary image events use the primary seed and
`preview_modes`; events also divisible by `milestone_every` add milestone modes
and `additional_seeds`. For example, sampling every 3 updates and expanding every
6 gives ordinary previews at 3 and 9, with expanded comparisons at 0, 6, and 12.
An expansion interval never creates an image event on its own. Leaving
`sample_every: 250` in a 12-update run would produce only the optional update-0
comparison, even if checkpoint and milestone settings are both 6.

### Checkpoint saves are independent

With `save.save_every: 6`, the two regular saves are **updates 6 and 12**. The
required initial and phase-boundary checkpoints add **updates 0, 2, and 10**.
Update 12 is also a boundary but is saved once. This gives **five complete
checkpoint directories: 0, 2, 6, 10, and 12**. Protected reference checkpoints can
make the retained count exceed `max_step_saves_to_keep`, which governs rolling
retention. Changing the image interval does not change this checkpoint schedule.

### Expected initial similarities

At initialization, the six learned-embedding modes should agree because both
LoRAs start at zero. The seventh base image can differ because it has no extra
suffix positions. Before the first A update at update 7, images 1 and 2 should
still agree, and images 3 and 4 should agree: the text adapter has not learned
anything yet. Similarity at update 6 is expected. The selected next visual sample
is update 12, after one A update, three further D updates, and two G updates. It
compares the final smoke state rather than isolating the immediate effect of the
A update. The separately saved update-10 checkpoint remains available for an
explicit later inference comparison.

Sampling at 1536 × 1024 for 28 steps makes this a substantial visual smoke. Image
generation and diagnostics can dominate its runtime even though training has
only 12 updates.

## The configuration groups in plain language

### Version and execution

- **`schema_version`, `spec_path`, `expected_spec_sha256`** identify the settings
  format and exact reference document. The SHA-256 value is a fingerprint that
  detects changed document bytes. They do not influence style strength.
- **`training_seed`** starts the training randomness: sample ordering, noise, and
  related choices. **`diagnostic_seed`** fixes probe selection and measurement
  samples without advancing the training randomness.
- **`deterministic_algorithms`** asks PyTorch to use deterministic operations;
  unsupported operations can error. A fixed seed alone does not guarantee
  identical results across different GPU/software combinations.
- **`encoder_gradient_checkpointing`** saves memory by recomputing intermediate
  encoder results during backpropagation. It costs extra compute. The native
  **`train.gradient_checkpointing`** provides the corresponding DiT control.
- **`dit_attention_backend`** chooses the native or flash attention path. It is
  an execution choice whose actual gradient behavior must be checked on the VM.
- **`error_policy: abort`** stops on a failed or nonfinite update instead of
  quietly skipping it and obscuring the experiment's history.

### Conditioning capacity

- **`num_tokens`** is the number of extra learned positions carrying the style
  signal. More positions provide more capacity and consume more token budget.
- **`initializer_text`, `initializer_jitter`, `initializer_seed`** choose the
  initial direction, the size of the small differences between positions, and
  the reproducible random draw for those differences. They set the starting
  point; subsequent A updates do the learning.
- **`adapter_rank`** controls the size of the small correction inside the text
  encoder. **`adapter_alpha`** sets its scale relative to that rank. Rank 4 and
  alpha 4 give an alpha/rank factor of one.
- **`overflow_policy: error`** requires the complete content prompt and suffix
  to fit. **`model.model_kwargs.max_text_length`** is their combined position
  limit, not a word count.

The original text encoder remains frozen even though its small Gen2 adapter
trains. Thus native `train_text_encoder: false` is consistent with A updates.
Text-embedding caches and encoder unloading are disabled because the suffix and
adapter need a fresh differentiable encoder path during training.

### Loss weights: what learning is encouraged to preserve

The main styled-image loss asks the active components to improve their denoising
prediction for a training image. The following values add specific constraints:

- **`neutral_weight`** penalizes drifting from the frozen original model when
  learned conditioning is absent. Higher weight puts more emphasis on preserving
  that neutral behavior during D and G updates.
- **`text_adapter_weight`** penalizes large encoder-adapter corrections relative
  to the original projection output at the suffix positions. It acts during A
  updates.
- **`gate_center_weight`** encourages gates to stay near one, the ordinary LoRA
  strength, during calibration.
- **`gate_smoothness_weight`** discourages abrupt changes in gate strength as the
  noise level changes.

These numbers multiply different measurements, so comparing their raw numerical
sizes does not show which constraint dominates. Inspect the recorded weighted
losses and gradients. A zero weight explicitly removes that constraint.

### Phase lengths and independent optimizers

**`warmup_updates`** trains D with initialized conditioning. During
**`refinement_updates`**, the two **`*_updates_per_cycle`** settings specify how
many D updates precede the A updates in each repeating cycle.
**`calibration_updates`** then freezes those components and learns only G. The
three stage lengths must sum to native `train.steps`.

Each of **`optimizers.diffusion`**, **`embedding`**, **`text_adapter`**, and
**`gates`** has the same controls:

- **`optimizer`** selects the native optimizer. `adamw8bit` is the reference
  choice for the VM; its state compression does not make every small state
  tensor or every model tensor eight-bit.
- **`lr`** is that component's learning rate: the scale of its update rule.
- **`optimizer_params`** passes additional native optimizer options.
- **`lr_scheduler`** chooses how the learning rate changes over updates, and
  **`lr_scheduler_params`** supplies its options. `constant` leaves it unchanged.

A `null` scalar inherits the corresponding `train` value. Option maps merge
over a copy of the shared map: `{}` inherits its options rather than clearing
them. Each scheduler advances only when its own optimizer updates. A component
with zero updates gets no constructed scheduler and is saved as explicitly
inactive.

#### Keeping the deliberate `weight_decay: 0.999`

The reviewed setting is deliberate and can be retained. With empty per-role
overrides, it applies to **all four AdamW optimizers**. It pulls their raw
parameters toward zero, scaled by each component's learning rate. In the usual
AdamW decay term, before accounting for the gradient update:

```text
parameter after decay = parameter before decay × (1 − learning_rate × weight_decay)
```

Thus 0.999 does not mean removing 99.9% each step. At the reviewed rates, the
direct per-step shrink is approximately 0.007992% for diffusion, 0.0999% for the
embedding and gates, and 0.000999% for the text adapter. These steps occur on each
component's local clock. Gradient updates act alongside the decay.

The output effect differs between components. Learned embedding vectors are
normalized to fixed lengths when used, so shrinking their raw vectors does not
directly shrink the forward embedding signal. Gate parameters pass through a
bounded `tanh` curve centered at one: shrinking those parameters tends to pull
the gates toward one, rather than directly multiplying gate outputs by the decay
factor. At zero initialization, gate weight decay alone changes nothing. For a
diffusion or encoder LoRA, both learned factors participate in its residual, so
raw parameter shrink and visible output strength are also different quantities.

Weight decay can help generalization, but a coefficient that works well for a
diffusion LoRA is not automatically equally useful for these other parameter
families. Keep the deliberate setting as a recorded choice and assess it from
the results. If a later comparison needs different decay for one family, set
that family's `optimizer_params.weight_decay` without changing the others.

### Gate shape and inference behavior

- **`gates.amplitude`** limits how far each learned gate may move from one. With
  0.5, the mathematical range is between 0.5 and 1.5. Each actual diffusion block
  has a smooth curve across the noise level.
- **`regularization_grid_points`** chooses how many fixed noise levels are used
  to measure the gate penalties and grid summaries. It does not add training
  updates or image-generation steps.
- **`missing_trigger_policy`** governs ordinary inference without an explicit
  diagnostic mode. `learned_neutral` uses the diffusion LoRA with ordinary
  conditioning; `base_bypass` bypasses the learned components.
- **`inference.lora_strength`** scales the conditional diffusion LoRA at
  inference only. Zero removes that residual while a styled route still retains
  its learned conditioning. The `base` mode also removes that conditioning.

### Dataset and native resource settings

- **`content_groups_file`** supplies optional labels for interpreting results,
  such as known content categories. It does not change sampling or assign
  quality judgments automatically.
- **`reject_train_validation_duplicates`** uses image hashes to prevent a
  training image from being mislabeled as held-out validation.
- **`resolution: [256, 768, 1280]`** creates native dataset variants for those
  resolutions. It is not a progression from low to high resolution. In a short
  run, not every resolution is guaranteed to appear.
- **`batch_size: 2`** with **`gradient_accumulation_steps: 1`** normally uses two
  examples per update; any smaller native batch is recorded with its actual
  size. Increasing accumulation combines more microbatches before updating.
- **`cache_latents_to_disk`** stores reusable VAE-encoded images. It saves future
  image-encoding work and uses disk space. **`num_workers: 0`** uses the reference
  loader replay mode for resume checks.
- **`quantize`** and **`quantize_te`** separately quantize the frozen diffusion
  model and text encoder. They can reduce weight memory. With
  `quantize_te: true`, real A-step gradients through the quantized encoder are
  part of what the VM smoke must verify. Quantization and checkpointing do not
  establish that every resolution/batch combination fits available VRAM.

### Numerical diagnostics

These measurements explain component behavior alongside the PNGs; they do not
assign an automatic visual-quality score.

- **`enabled: true`** keeps core recording active.
- **`activation_every`** controls how often internal signal and adapter-residual
  summaries are collected. **`time_bin_edges`** groups those measurements by
  noise level for comparison.
- **`gradient_probe_every`**, **`gradient_probe_examples`**, and
  **`gradient_probe_taus`** choose the interval, fixed example count, and noise
  levels for isolated measurements of how individual losses push parameters.
  Probes do not apply training updates.
- **`gradient_probe_max_coordinates`** caps the sampled parameter entries used
  for those comparisons. Zero requests all entries; a positive cap produces
  sampled estimates and limits memory.
- **`gate_log_every`** controls regular gate-curve records.
  **`spectra_every`** controls measurements of how the low-rank factors use their
  available directions. Boundary measurements provide additional references.
- **`full_update_norm_every`** requests exact before/after parameter-change
  measurements at an interval. Zero disables those full snapshots.
  **`parameter_sample_elements_per_family`** sets the fixed sample used for
  lighter update measurements.
- **`tensor_memory_budget_mb`** limits additional diagnostic tensor buffers. It
  is not a cap on the model's total VRAM consumption.
- **`prefix_atol`** and **`prefix_rtol`** are absolute and relative tolerances for
  checking that original prompt positions remain consistent with the frozen
  encoder reference. A failure needs investigation; it is not a style-strength
  setting.
- **`probes.every`**, **`num_examples`**, **`source`**, and **`taus`** configure
  fixed denoising comparisons with diffusion and conditioning enabled/disabled.
  `source: training` labels them as training probes; validation needs actual
  held-out examples. **`save_fixed_latent_packet`** preserves their images in
  latent form, noise, and other replay inputs.
- **`tensor_dumps.enabled`** optionally saves larger raw tensor packets. The
  allowed **`names`** select their contents; **`max_packets`** and
  **`max_total_mb`** limit retained storage. Core scalar measurements remain even
  when tensor dumps are off.

### Recording, images, and recovery

- **`recording.flush_every_updates`** controls periodic writer flushes. Saves,
  boundaries, and errors also flush. **`rotate_mb`** splits growing JSONL streams
  into segments; **`compression`** can gzip closed segments.
- **`writer_queue_records`** bounds the record queue. Training waits when needed
  rather than silently dropping core records. **`max_core_recording_mb`** is a
  hard budget for core records, separate from model weights, images, and optional
  tensor dumps. Reaching it stops the run instead of continuing without records.
- **`sample.sample_every`** sets the repeating image interval, independently of
  saves and phase boundaries. **`sample_start_step`** controls when repeating
  samples become eligible. **`train.skip_first_sample`** separately controls the
  optional initialization comparison. No extra image is forced at the final
  update when it is not a scheduled image event.
- **`evaluation.milestone_every`** expands an already scheduled image event when
  its update is also divisible by this value. It does not trigger images by
  itself. **`preview_modes`** selects the regular set, and **`milestone_modes`**
  adds the expanded set at those events; duplicate requests are combined.
- **`additional_seeds`** adds matched seeds at expanded events; ordinary previews
  use `sample.seed`. **`prompt_groups`**
  labels existing prompt IDs such as `p000`; it does not add prompts.
  **`make_contact_sheets`** produces labeled overview sheets while preserving
  full-size individual images and their metadata.
- Native **`sample_steps`**, **`guidance_scale`**, **`width`**, and **`height`**
  control image-generation work and appearance. Hold them constant within each
  comparison. **`walk_seed: false`** preserves matched seeds.
- **`resume_from`** points to a complete Gen2 checkpoint directory.
  **`strict_resume`** checks that the saved training contract still matches.
  **`save_at_stage_boundaries`**, **`protect_stage_checkpoints`**, and
  **`save_initial_state`** retain interpretable recovery/reference points.
  Native **`save_every`** adds interval checkpoints, and
  **`max_step_saves_to_keep`** controls rolling retention without removing the
  protected boundary checkpoints.
- Native **`save.dtype`** controls inference exports. Complete checkpoints keep
  float32 training masters and optimizer state for resume. A bare diffusion LoRA
  cannot reproduce the complete pipeline because it lacks the learned embedding,
  encoder adapter, and gates.

## What remains to be established on the VM

Configuration parsing, prompt token counting, and local tests are separate from
real-model acceptance. The RTX PRO 6000 VM run still needs to establish memory
fit, the installed quantizer/optimizer gradient paths, finite D/A/G updates, and
the resulting image comparisons. Twelve successful updates establish a useful
execution smoke; a longer run is needed to judge whether the learned components
achieve the intended style and which unconditional setting is preferable.

Preserve the individual PNGs with their JSON metadata, the labeled contact
sheets, and numerical diagnostics. They connect a visual result to its exact
checkpoint, seed, component switches, and unconditional strength. The original
repository-root `gen2/` directory remains a read-only source for this pipeline;
run outputs are written beneath the configured training run directory.
