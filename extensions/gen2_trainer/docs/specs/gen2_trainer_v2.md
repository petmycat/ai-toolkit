# Gen2 Trainer v2 — Standalone Activator Design

**Version:** 2.0.0  
**Date:** 2026-09-17  
**Status:** Agreed design, recorded for user review; implementation awaits a separate user instruction.  
**Initial backend:** Ideogram 4.0 through this ai-toolkit checkout.  
**Primary experiment:** In-place learned token vectors, with the text encoder and image generator frozen.  
**Reference location:** `F:/ai-toolkit/gen2/gen2_trainer_v2.md`.

## 0. Authority, preservation, and implementation boundary

This document records the v2 proposal accepted by the user in the conversation. It is a design reference, not a report of an implemented or successful v2 experiment.

The user authorized creating this file under `gen2/` and marking this file read-only. That authorization is a specific exception to the existing rule that `F:/ai-toolkit/gen2/` is read-only to the coding agent. It does not authorize changes to other files in that directory or authorize implementation now.

Preserve this reference during implementation. Record implementation notes, compatibility findings, experimental results, and proposed design changes separately. A change to the scientific mechanism must be discussed with the user rather than silently substituted for this design. Routine implementation decisions that preserve the design do not require another design discussion.

When implementation is authorized, pipeline code belongs under **`extensions/gen2_trainer/` and must be tracked by Git**. The local `gen2/` directory is intentionally ignored by Git and is used for user-provided specifications and experimental results. Do not change that ignore policy as a side effect of creating this reference. Preserve the v1 reference and existing experimental results.

For the v2 experiment, this document supersedes v1 requirements that conflict with the scope below. In particular, v1's diffusion/conditioning/gate phase schedule, learned suffix placement, fixed token-radius constraint, and original-caption-feature invariance are not v2 requirements.

## 1. Objective and roadmap

### 1.1 The user's target

The user's observations and goals are qualitative judgments of intended style fidelity, not calibrated numerical metrics:

- An ordinary diffusion LoRA with an arbitrary trigger produced approximately 0% of the intended style in the difficult prompt.
- The base model with the named phrase "Ghibli anime" produced approximately 50%.
- The ordinary LoRA with that named phrase produced approximately 75%.
- The standalone activator should make the base model perform **at least as well as the named phrase**, preferably better.
- The later LoRA-plus-activator system aims for a substantially stronger result, described by the user as greater than 90%.

The old ordinary LoRA checkpoint is unavailable. The current v1 LoRA is not considered a useful recoverable style checkpoint by the user. Neither checkpoint is a dependency for v2.

The intended activator must work with unnamed styles. Training must not depend on the user discovering a successful style name, phrase, or description. The named phrase available for this particular dataset is an **evaluation benchmark only**.

The activator must remain useful when combined with content descriptions. For example, it should help the base model draw a tiger, a dining room, or an explosion in the dataset's style, rather than merely recreate objects or layouts from the training images.

### 1.2 Separate the roadmap into objectives

- **V2:** Establish a standalone activator on the frozen base model.
- **V3:** Train a diffusion LoRA with the successful activator enabled.
- **V4:** Investigate timestep- and block-dependent application of diffusion LoRA residuals.

Do not proceed to v3 merely because v2 executes successfully or reduces training loss. V2 must meet the visual acceptance criterion in Section 11.

### 1.3 Why v2 differs materially from v1

V1 trained its conditioning module while the diffusion LoRA was active. The learned conditioning and diffusion changes could therefore develop a relationship that worked only together. Its conditioning objective did not directly train the requirement that the activator work on the untouched base model.

V2 keeps the image-generating model frozen throughout training. Any improvement obtained by optimization must come through the learned conditioning. Every optimizer update in the primary experiment trains that standalone activator.

This removes one source of ambiguity. It does not guarantee that reconstruction training will discover a reusable style representation rather than a dataset-specific shortcut.

## 2. Research basis and limits

**Textual Inversion** demonstrates learning new object or style token embeddings while keeping the text encoder and image generator frozen. Its original implementation initialized embeddings from coarse descriptive words. It supports the general approach, but does not establish that random initialization is equally reliable on Ideogram/Qwen. [Textual Inversion paper](https://arxiv.org/html/2208.01618)

**DreamStyler** finds that descriptions of reference-image content help disentangle style from objects and scene composition. It also reports limitations in ordinary textual inversion and improvements from embeddings that vary across denoising stages. Those results support retaining the user's detailed captions and taking representation limits seriously. They do not justify importing stage-dependent conditioning into the initial v2 experiment or claiming its reported performance for this different architecture. [DreamStyler paper](https://arxiv.org/html/2309.06933)

Official Diffusers textual-inversion tooling supports multiple learned vectors associated with a placeholder token. Its documentation and implementation are useful engineering references, not validated Ideogram hyperparameter recipes. [Diffusers textual-inversion guide](https://huggingface.co/docs/diffusers/main/training/text_inversion) · [Official training implementation](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py)

The known phrase demonstrates that the frozen model has access to some relevant visual knowledge. Finding an equally effective learned condition from images and random initialization remains an optimization hypothesis. Matching the phrase is an acceptance criterion, not an assumed outcome.

## 3. Primary architecture and parameter ownership

The default v2 experiment trains only a shared bank of learned input token vectors.

### 3.1 Trainable component

- A bank `E = [E1, E2, E3, E4]` of four learned vectors.
- Each vector has the native Qwen input embedding dimension.
- The inspected Qwen encoder has dimension 4096, giving **16,384 trainable values** for four vectors.
- Trainable master parameters use FP32.
- The same bank is shared across all images, captions, and occurrences of the trigger within one style package.

### 3.2 Frozen components

- Original Ideogram conditional diffusion transformer.
- Original Qwen text encoder, including all ordinary vocabulary embeddings.
- VAE.
- Original unconditional transformer used for sampling.

The primary experiment constructs no trainable diffusion LoRA, text-encoder adapter, or block/timestep gates. It has no diffusion warm-up or alternating D/A/G optimization phases. Do not create inactive optimizers or schedulers for components that never train.

A text-encoder adapter is a separately evaluated capacity extension described in Section 12. It is not silently enabled in the default experiment.

### 3.3 What the activator represents

The user-visible trigger is an identifier for loading and invoking the learned bank. Its arbitrary spelling is not itself the style representation.

The learned vectors have a distinct token identity from existing vocabulary words. Their numerical meaning may approach useful existing style concepts. There is no penalty requiring them to remain far from the embeddings or features associated with a known phrase. Discovering useful semantic relationships is part of the intended mechanism.

## 4. In-place trigger compilation and contextual interaction

### 4.1 Placement

Replace each `[trigger]` occurrence, or occurrence of the configured reserved literal trigger, with the ordered learned bank at that occurrence's location **before Qwen processes the prompt**.

Conceptually:

```text
A tiger in [trigger] style
              |
              v
A tiger in [E1 E2 E3 E4] style
```

The symbols above denote input vectors, not literal text sent to Qwen.

Retain Ideogram's native caption normalization and chat template. "Original position" means the corresponding location in the native normalized caption. The native JSON normalizer can reorder fields; v2 must not bypass that processing merely to preserve source-file byte offsets.

Do not remove the markers and append a learned suffix after the caption or after the chat generation marker. The user's placement preference is settled; another suffix-versus-original-position experiment is not a prerequisite for v2.

### 4.2 Repeated occurrences

Inspection of the recorded dataset found 40 structured captions, each containing three `[trigger]` occurrences: one each in `style_description.aesthetics`, `style_description.medium`, and `style_description.art_style`.

Preserve all occurrences. Three occurrences of a four-vector bank produce:

```text
... E1 E2 E3 E4 ... E1 E2 E3 E4 ... E1 E2 E3 E4 ...
```

This is **12 learned input positions using one shared set of four vectors**, not three independently trained style identities. In general, `n` occurrences use `n * M` positions for a bank of `M` vectors.

The contextual Qwen features at repeated occurrences may differ because their positions and preceding text differ. Training and sampling must follow the same expansion policy. Record occurrence counts and insertion spans so placement cannot silently drift.

### 4.3 Preserve the native text path

The implementation must preserve ordinary tokenization around the reserved marker, native attention masks, consecutive positions, chat serialization, feature taps, and feature packing.

Use an explicit reserved-marker compilation mechanism; do not let arbitrary trigger spelling accidentally become ordinary subword tokens in the learned-activator path. Avoid fragment-by-fragment tokenization that silently changes ordinary text boundaries. An isolated tokenizer copy with reserved internal placeholders and differentiable input-embedding replacement is a suitable implementation approach, provided it preserves the native baseline path and is tested.

Training captions missing the marker must be reported during preflight rather than silently receiving an appended activator. Verify that caption normalization preserves the intended occurrences.

### 4.4 Why the activator can interact with content

The input vectors are shared, but Qwen's resulting features depend on context. Ordinary text positions after the inserted vectors can attend to them through the native causal encoder. Ideogram receives the resulting contextualized sequence and performs its native image/text interaction.

Consequently, activator-plus-tiger and activator-plus-room can supply different content conditioning while sharing learned stylistic information.

Freeze the text encoder's **weights**, not the computed features of ordinary words. V1's requirement that all original caption features remain unchanged is explicitly removed. Do not detach downstream ordinary-token features or mask away their contribution to activator gradients.

When the marker is absent, use the ordinary native base-model text path. No activator or optional adapter should affect that path.

## 5. Random initialization and vector magnitude

Initialize the four vectors independently using a reproducible recorded random seed.

Sample initial values using ordinary Qwen vocabulary embedding statistics, and match initial vector magnitudes to typical ordinary vocabulary embeddings. This is numerical calibration, not semantic initialization from a selected word or phrase. The implementation must document and record the exact sampling/statistics procedure for reproducibility.

Do not use "Ghibli anime," another known style description, or a generic style word to initialize the bank. Do not average the arbitrary trigger string's token embeddings as the default v2 initialization.

Use FP32 master vectors. After initialization, allow both their direction and magnitude to change. Remove v1's mandatory projection onto a sphere of fixed initial radius.

Monitor:

- Per-vector norm and movement from initialization.
- Similarities between learned vectors.
- Gradient magnitude and actual optimizer update magnitude.
- Nonfinite values and abnormal conditioning activation growth.

Free vector magnitude is an initial design choice, not a claim that unconstrained embeddings always generalize better. Do not silently introduce normalization, clipping of vector norms, or an additional embedding penalty as a replacement mechanism if the experiment is difficult.

## 6. Training objective and gradient path

### 6.1 Inputs and native flow target

For a caption `q` and its image, let:

- `z0` be the image latent produced by the frozen VAE using the native Ideogram conventions.
- `epsilon` be sampled Gaussian noise.
- `tau` be the noise fraction, with 0 meaning clean and 1 meaning noise.
- `zt = (1 - tau) * z0 + tau * epsilon`.
- `C_frozen(q; E)` be native Qwen conditioning with the learned vectors inserted in place.
- `F_frozen` be the frozen Ideogram conditional velocity predictor, including the native output conversion.

The target is `epsilon - z0`. The primary objective is:

```text
L(E) = mean || F_frozen(zt, tau, C_frozen(q; E)) - (epsilon - z0) ||^2
```

Use the native full noise-time range with uniform sampling over the existing training timestep schedule for this first experiment. Preserve native latent scaling, prediction conversion, and valid-element loss reduction.

There is one training objective and one active parameter family in the primary experiment. Every committed optimizer update trains the activator.

### 6.2 Plain-language interpretation

Show the frozen model noisy versions of the training images and their captions. Adjust the learned vectors so the model better predicts how to recover the images from noise.

The detailed human captions already explain much of the scene content. They give the activator an opportunity to represent the shared appearance that remains insufficiently described. Reconstruction loss does not itself label which visual improvements are style, so held-out visual behavior remains essential.

### 6.3 Gradient ownership

Frozen Qwen and diffusion weights have `requires_grad=False`, but their forward computations remain differentiable with respect to their inputs. Do not wrap these complete conditioning-to-prediction computations in `no_grad`, and do not detach the features that connect the prediction to the learned vectors.

Backpropagation must traverse:

```text
loss -> frozen diffusion computation -> Qwen features
     -> frozen Qwen computation -> learned input vectors
```

The VAE can run without gradients because image encoding prepares the target and does not lie on the trainable conditioning path.

Reuse gradient-safe checkpointing and native model operations. Checkpoint recomputation must preserve the same token positions, masks, and execution context as the original forward pass.

### 6.4 Additional objectives are outside the initial experiment

The initial run does not add a style-name feature-matching target, CLIP style reward, teacher-generated training images, neutral-preservation loss, new timestep weighting, or a CFG-space training loss.

The known phrase must not appear as training-caption substitution, initialization, teacher conditioning, or an optimization target. It remains available for the fixed visual benchmark.

Do not treat a reconstruction-loss decrease or a difference from the neutral model as proof of successful style learning.

## 7. Caption length, data integrity, and caching

Keep the human-written source captions intact. Do not manually or automatically rewrite them into shorter descriptions. Continue the user's approved token-truncation approach.

The initial total token budget is **3072**, accounting for the native chat wrapper, ordinary caption tokens, and every expanded learned-token occurrence.

For overlong inputs:

1. Determine the complete expanded length and all learned-token spans before truncation.
2. Shorten the tokenized caption content from its tail to fit the budget.
3. Preserve all learned-token spans and the native chat-template boundaries.
4. If those requirements cannot be satisfied, report the specific input before expensive model initialization instead of silently dropping an activator or relocating it within the caption. Preservation refers to its semantic insertion location and order; record the actual numerical positions after expansion and truncation.
5. Record original length, resulting length, occurrence count, and number of omitted content tokens.

This is runtime token truncation, not an edit to caption files. The resulting caption body may end partway through a structured description; do not claim its complete original meaning remains available after truncation.

Use the same declared compilation and truncation policy for training and sampling. For an overlong evaluation prompt, select a common retained ordinary-caption prefix that fits all comparison modes, accounting for their different replacement lengths. Use that same ordinary content for the marker-removed, named-phrase, initial-token, and learned-token modes. Accidental differences in retained scene content must not masquerade as an activator benefit.

Image latents may remain cached because the VAE is frozen. Learned text features must be recomputed while the vectors are training. Tokenization and other genuinely fixed preprocessing may be cached when valid for the resolved configuration.

## 8. Initial optimizer and pilot settings

These are **starting settings for a bounded pilot**, not established optimal values for Ideogram.

- Number of learned vectors: **4**.
- Initial budget: **500 committed activator optimizer updates**.
- Effective batch size: **4 examples**, using gradient accumulation as needed.
- Optimizer: ordinary **AdamW** with FP32 optimizer state.
- Learning rate: **5e-4**.
- Epsilon: **1e-8**.
- Betas: **(0.9, 0.999)**.
- Weight decay: **0**.
- Learning-rate schedule: **constant** for the initial pilot.
- Sampling: initialization and every **100 committed updates**, yielding 0, 100, 200, 300, 400, and 500.

Do not silently scale the learning rate with batch size. Keep the distinction between microbatches and committed optimizer updates explicit in logs, sampling, and checkpoint counters.

Ordinary AdamW is recommended because the learned component is very small. Four 4096-dimensional FP32 vectors occupy approximately 64 KiB; their two FP32 optimizer moment tensors occupy approximately 128 KiB. This small optimizer state is not the main VRAM cost.

AdamW8bit should remain a supported standard optimizer option when verified for this path. Its memory benefit is minor for the token-only experiment. Do not carry forward a requirement to support every unusual optimizer before evaluating the hypothesis.

The native optimizer factory currently supplies a default epsilon of 1e-6. V2 must deliberately resolve the requested value, assert the effective optimizer parameter-group settings, and record them. Merely adding an unsupported duplicate keyword to `optimizer_params` is not a valid implementation. Preserve actual optimizer settings and state on resume.

Zero weight decay is the recommended pilot setting for these standalone learned vectors. Generalization must be judged from visual behavior rather than assumed to follow from shrinking their magnitudes.

## 9. Memory plan for the 96 GB VM

The intended training machine is the user's RTX PRO 6000 VM with **96 GB VRAM** and ai-toolkit dependencies installed. The user will run the configurations manually and provide results.

The design does **not guarantee** that a particular physical batch size, image resolution, and caption length fit within 96 GB. Frozen parameters still participate in the backward computation needed to reach the input vectors. Long Qwen sequences and high-resolution diffusion activations can dominate memory.

### 9.1 Separate physical batch from effective batch

The target effective batch size of four does not require four images to be resident simultaneously.

A memory-conservative starting configuration is:

```text
microbatch size = 1
gradient accumulation = 4
effective batch size = 4
```

If a larger physical microbatch fits, accumulation can be reduced while preserving the effective batch. The user explicitly accepts reducing physical batch size if necessary.

Accumulation must scale the loss correctly and perform one optimizer step per complete update. Do not change the meaning of 500 updates or the sampling schedule when changing the physical microbatch.

### 9.2 Other memory considerations

- Use gradient-safe activation checkpointing through frozen Qwen and the conditional diffusion model.
- Preserve FP32 trainable masters without forcing all frozen-model computation to FP32.
- Reuse appropriate native precision, quantization, and memory-management support when the input-gradient path is valid.
- Record the actual model precision and use the same precision for learned-activator and named-phrase comparisons.
- Cache image latents where appropriate.
- Account separately for sampling memory, including the original unconditional model.
- Log peak allocated/reserved VRAM for representative forward/backward and sampling work.

Check the intended long-caption and high-resolution cases during the mechanical smoke, not only the smallest bucket. If microbatch size one still fails, inspect sequence/resolution/activation and model residency costs. Do not silently lower caption limits, drop high-resolution data, change the objective, skip failed batches, or count an OOM as a completed optimizer update.

## 10. Sampling and unconditional-model behavior

Use the **original frozen unconditional transformer** for sampling, with ordinary CFG. The learned activator supplies conditional text features. The unconditional model receives no learned activator and remains the same across comparison modes.

There is no diffusion LoRA in the primary v2 experiment, so conditional-versus-unconditional diffusion LoRA strength sweeps do not apply here. Preserve that research question for the later diffusion-LoRA work.

Keep the user's existing explosion prompt as the main routine diagnostic prompt. Preserve its human-written content and original trigger locations.

For fixed seeds, compare:

1. Base model with the trigger removed.
2. Base model with "Ghibli anime" substituted at the corresponding trigger positions.
3. Base model with the initial random activator.
4. Base model with the current trained activator.

The first three are fixed reference modes and need only be generated once for an unchanged evaluation configuration. Later sample events can generate the current activator output and present it alongside the references. At update zero, the initial and current bank are identical and should be labeled accordingly rather than presented as independent learned results.

Use a small fixed set of seeds, preferably at least two, with identical sampler, CFG, resolution, base weights, and precision across modes. Record the resolved prompt, insertion positions, seed, model identities, and sampling settings.

Sampling timing is independent of checkpoint-save timing. Print visible progress, including update, comparison mode, seed/image count, and denoising progress, so image generation does not appear stuck.

## 11. Acceptance, diagnostics, and stopping decisions

### 11.1 Mechanical correctness is a prerequisite

Before treating an experiment as evidence, verify:

- Learned vectors receive finite, nonzero gradients and actual optimizer updates.
- Only the intended trainable parameters receive updates and optimizer state.
- Frozen Qwen, vocabulary, conditional diffusion, VAE, and unconditional weights remain unchanged.
- Marker occurrences, insertion spans, expanded lengths, and truncation are correct and recorded.
- Native attention masks, positions, feature extraction, feature packing, and velocity conversion are preserved.
- Gradients can reach the bank through affected ordinary text features as well as through the inserted positions themselves.
- Without the marker, the base-model path matches the native path to measured numerical tolerance.
- Saving and reloading the activator reproduces its conditioned predictions under controlled inputs.
- Gradient accumulation and update counting behave correctly.

Do not retain the v1 assertion that every original caption feature must equal its marker-free counterpart. Such equality would conflict with the intended in-place contextual interaction.

### 11.2 Useful numerical records

Record training loss, sampled noise level, token gradient/update magnitudes, vector norms and movement, learning rate and actual optimizer settings, truncation counts, timing, and memory use. Noise-level summaries can help reveal uneven behavior without changing the training objective.

Finite values, nonzero gradients, changed conditioning, changed pixels, and lower reconstruction loss establish useful facts about execution. None establishes visual success by itself.

### 11.3 Visual pass criterion

The minimum success criterion is:

> The trained activator on the frozen base model expresses the intended dataset style at least as well as the named phrase, while preserving the requested scene and objects.

The user's approximately 50% named-phrase result is a qualitative benchmark, not a numeric threshold to manufacture from MSE, embedding similarity, or an unrelated image metric.

Assess style and content preservation separately. Reject apparent style gains obtained by replacing requested subjects with training-image subjects, copying layouts, creating artifacts, or materially losing prompt adherence. A single lucky seed is insufficient evidence.

Once the main explosion prompt shows useful progress, perform a small acceptance check using **two additional subjects absent from the training captions**, with a few fixed seeds. This checks the compositional goal exemplified by "a tiger in this style." The existing single prompt remains the routine sampling prompt. Success on that prompt alone establishes success on that prompt, not broad generalization.

### 11.4 Bounded pilot and follow-up

The first 500 activator updates are a review point. Do not automatically extend the experiment to another long run merely because it completes or the loss decreases.

Stop and investigate ineffective gradients/updates, nonfinite values, collapsed outputs, severe content leakage, or deteriorating prompt adherence. If updates function but visuals show no useful trend, review representation and optimization before granting more training time.

A failed pilot means that this representation, initialization, objective, and tested budget did not meet the target. It is not proof that every possible activator is impossible. Conversely, plausible alternatives are not a reason to keep spending full-run budgets without a concrete next hypothesis.

V3 begins only after the standalone activator meets the agreed visual requirement.

## 12. Optional capacity extension: token bank plus Qwen adapter

This is a separately evaluated extension within the standalone-activator objective. It is **not enabled in the primary v2 experiment**. Decide whether to pursue it from the token-only result and discuss the change before treating it as the new default.

If pursued, the proposed mechanism is:

- A small rank-4 LoRA adapter on Qwen decoder MLP `down_proj` layers.
- Original Qwen and diffusion weights remain frozen.
- The adapter and learned token bank train together against the same standalone base-model reconstruction objective.
- For an example invoking the activator, apply the adapter to all valid text positions, not only learned positions.
- For an example without the marker, bypass the adapter completely and retain the native base route.
- Exclude padding positions from adapter effects.

This gives the style package more capacity to modify how the entire prompted scene is encoded. It also increases the opportunity to memorize training content or let the adapter carry most of the effect.

Retain comparisons that distinguish these possibilities:

- Native base and named-phrase references.
- Learned tokens with the adapter off.
- Initial tokens with the adapter on.
- Learned tokens with the adapter on.

If this extension succeeds, describe the deliverable as a **token-plus-text-adapter package**. Do not describe its performance as evidence that the learned token bank alone succeeds. Its additional optimizer settings require their own recorded pilot choice; the primary token-only settings do not silently define them.

Timestep-dependent tokens, block-specific conditioning, and diffusion residual gates remain outside the initial v2 experiment.

## 13. Checkpoints, inference package, and reproducibility

Keep storage small and sampling independent from saving.

For the primary experiment, save:

- Learned token vectors and their initialization state/provenance.
- Model and tokenizer identities, trigger semantics, placement and repetition policy, feature extraction conventions, truncation policy, precision, and resolved configuration.
- The specification identity/hash in implementation and run provenance.

Resume state additionally includes the active optimizer state, any active scheduler state, RNG state, data position, and committed update counters. Do not package frozen base weights, diffusion adapters, gates, or their optimizer/diagnostic state.

Maintain **one rolling resume checkpoint and a final inference export** as the initial retention policy. Keep lightweight logs and the fixed visual references; avoid retaining a large tree of redundant intermediate checkpoints. Any optional user-selected milestone export should be explicit.

The raw learned vectors occupy approximately 64 KiB in FP32; metadata and resume state add overhead. The small artifact does not imply low training activation memory.

Loading an inference package must restore the same in-place replacement semantics and native conditioning behavior used during training. A generic loader that silently appends the vectors as a suffix is not compatible with this design.

## 14. Summary of the agreed initial experiment

1. Learn four independent random input vectors, shared across original trigger occurrences.
2. Freeze the native text encoder, vocabulary, diffusion model, VAE, and sampling unconditional model.
3. Train through the complete differentiable conditioning path using the native conditional flow-matching objective.
4. Preserve human captions, with explicitly reported tail-token truncation under the initial 3072-token budget and protection for markers and chat boundaries.
5. Start with a 500-update pilot, effective batch four, AdamW at 5e-4 with epsilon 1e-8 and zero weight decay.
6. Use physical microbatch one plus accumulation four as a conservative memory starting point; measure actual VRAM use on the 96 GB VM.
7. Sample independently at initialization and every 100 updates, keeping fixed native-base, named-phrase, and random-initialization references.
8. Require visual parity with or improvement over the named phrase, together with content preservation, before advancing to v3.
9. Consider a trigger-enabled Qwen adapter only as a separately evaluated capacity extension within v2.
10. Keep implementation tracked under `extensions/gen2_trainer/`, preserve this read-only reference, and await the user's separate implementation instruction.
