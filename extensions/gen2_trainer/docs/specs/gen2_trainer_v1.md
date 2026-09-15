# gen2_trainer — pipeline process specification v1

**Status:** immutable implementation reference.  
**Version:** `1.0.0` (the user's pipeline v1).  
**Date:** 2026-09-11.  
**Process identifier:** `gen2_trainer`.  
**Initial backend:** Ideogram 4.0 through ai-toolkit.  
**Audience:** the coding agent implementing the pipeline; the user and research assistant interpreting subsequent experiments.

## 0. Authority, immutability, and how to handle questions

This file is the implementation contract. Read it completely before designing the implementation. **Do not modify this file during implementation, including to fix typos, clarify equations, record decisions, or reflect code changes.** Preserve its original bytes. Compute its SHA-256 before implementation; record that hash in the implementation manifest, every training run manifest, and every checkpoint. Add a test that detects changes to the reference file. Keep implementation notes, questions, answers, compatibility findings, and deviations in separate files. A subsequent authorized design revision requires a new versioned specification; it must not overwrite this v1 reference.

The latest descriptor-free design in this file supersedes the earlier discussion draft that required a descriptive style anchor. Neither the earlier DOCX proposal nor an inferred intent overrides an explicit requirement here.

**Ask the user whenever you are genuinely unsure about a requirement, encounter an unresolved contradiction, need to change a mathematical mechanism, or need a consequential decision.** State the exact ambiguity, relevant specification section, implementation evidence, and smallest decision needed. Continue independent, unblocked work. Do not silently invent a substitute loss, token location, gradient rule, branch behavior, optimizer restriction, architecture, or training schedule. Record confirmed answers separately with their provenance. If an answer changes this mathematical contract, request an authorized successor specification before implementing that change as the default.

Routine code organization may vary when it preserves this contract. Prefer an existing upstream implementation over a new equivalent one. If exact reuse is impossible, explain why, identify the smallest new adapter or orchestration code required, and record the boundary. Implementation correctness does not establish that the scientific hypothesis succeeds.

## 1. Problem, intended mechanism, and scope

The user is an experienced trainer of image-generation LoRAs. Their observed problem is that a style LoRA can show substantial target-style capability under some prompts but weak styling under others despite the same trigger. On one dataset, a known style description rescues some failures only when the LoRA is present. This motivates investigating conditioning-dependent behavior; it does not prove that the complete target style is recoverable for all scenes.

The pipeline must work with unnamed styles. Required training inputs are images, content captions, and a user-selected arbitrary trigger identifier. **Do not require a style name, an accurate literal style description, manual nearest-style search, a vision-language description generator, or the user's old LoRA.** The old LoRA is inaccessible as weights and is not a dependency.

V1 trains a fresh diffusion LoRA and a dataset-specific continuous style-token bank with a masked text-encoder adapter. The style representation is learned from image reconstruction. A neutral preservation branch explicitly discourages the loaded diffusion LoRA from changing base-model predictions when the learned style condition is absent. Alternating updates separate diffusion and conditioning optimization. A final stage calibrates smooth, bounded per-block timestep gates with both LoRA weights and conditioning frozen.

V1 is one style package per run, an Ideogram text-to-image backend, and a single training process on one GPU, with native supported memory-management facilities where gradient-safe. It is not a multi-style meta-training system, an image-reference inference adapter, an image editor, a multi-GPU implementation, or a new base model. Structure backend interfaces for later ports, but reject unsupported architectures and `WORLD_SIZE > 1` clearly rather than silently using an unvalidated path. All optimizers supported by the installed ai-toolkit factory must remain selectable; this requirement is independent of the single-process execution scope.

A successful run should improve difficult and unseen-content prompts while preserving content fulfillment. A working trigger parser, a falling reconstruction loss, a large residual, or a nonzero gradient is insufficient evidence of that outcome.

## 2. Upstream baseline and extension architecture

### 2.1 Verified upstream reference

The integration points below were inspected at ai-toolkit revision **`f56b5a1d405f819c74724228564e99982624c186`**, dated 2026-09-10. This is a reference snapshot, not an instruction to downgrade the user's checkout. Record the actual checkout, dependency versions, dirty diff hash, and compatibility findings. If a changed API has identical semantics, adapt the bridge. If its semantics conflict with this document, ask.

| Reuse area | Inspected upstream location | Required use or boundary |
|---|---|---|
| Process registration | [toolkit/extension.py](https://github.com/ostris/ai-toolkit/blob/f56b5a1d405f819c74724228564e99982624c186/toolkit/extension.py), [extensions/example](https://github.com/ostris/ai-toolkit/blob/f56b5a1d405f819c74724228564e99982624c186/extensions/example/__init__.py) | Register an `Extension` with `uid = "gen2_trainer"`, lazy `get_process()`, and `AI_TOOLKIT_EXTENSIONS`. The loader discovers both extension directories. |
| Trainer lifecycle and helpers | [BaseSDTrainProcess](https://github.com/ostris/ai-toolkit/blob/f56b5a1d405f819c74724228564e99982624c186/jobs/process/BaseSDTrainProcess.py), [SDTrainer](https://github.com/ostris/ai-toolkit/blob/f56b5a1d405f819c74724228564e99982624c186/extensions_built_in/sd_trainer/SDTrainer.py) | Reuse configuration, loading, data/latent preparation, logging, saving and sampling helpers where their contracts match. Gen2 owns phase selection, branch objectives, optimizer stepping and logical-update accounting. |
| Optimizers | [toolkit/optimizer.py](https://github.com/ostris/ai-toolkit/blob/f56b5a1d405f819c74724228564e99982624c186/toolkit/optimizer.py) | Call `get_optimizer(params, optimizer_type, learning_rate, optimizer_params)` for every parameter family. Do not implement or duplicate optimizer algorithms or maintain a smaller local name whitelist. |
| LR schedulers | [toolkit/scheduler.py](https://github.com/ostris/ai-toolkit/blob/f56b5a1d405f819c74724228564e99982624c186/toolkit/scheduler.py) | Call `get_lr_scheduler`; adapt its existing keyword conventions, including optimizer-local horizons. |
| Config types | [toolkit/config_modules.py](https://github.com/ostris/ai-toolkit/blob/f56b5a1d405f819c74724228564e99982624c186/toolkit/config_modules.py) | Reuse native model, network, train, dataset, save and sample schemas; add a separate strictly validated Gen2 configuration namespace. |
| Model loading and VAE | [ideogram4.py](https://github.com/ostris/ai-toolkit/blob/f56b5a1d405f819c74724228564e99982624c186/extensions_built_in/diffusion_models/ideogram4/ideogram4.py) | Reuse `Ideogram4Model`, checkpoint loading, quantization/placement support, latent normalization, VAE encoding/decoding and key conversion. |
| Packing, velocity and sampling utilities | [Ideogram pipeline](https://github.com/ostris/ai-toolkit/blob/f56b5a1d405f819c74724228564e99982624c186/extensions_built_in/diffusion_models/ideogram4/src/pipeline.py) | Reuse `pad_text_features`, `predict_velocity`, patchification, and `get_ideogram4_sigmas`. Gen2 adds differentiable suffix extraction and explicit inference branch control. |
| Diffusion modules | [Ideogram transformer](https://github.com/ostris/ai-toolkit/blob/f56b5a1d405f819c74724228564e99982624c186/extensions_built_in/diffusion_models/ideogram4/src/transformer.py) | Reuse the transformer and attention implementations unchanged. Its native checkpointing is non-reentrant in the inspected source. |
| Low-rank adapters | [lora_special.py](https://github.com/ostris/ai-toolkit/blob/f56b5a1d405f819c74724228564e99982624c186/toolkit/lora_special.py), [network_mixins.py](https://github.com/ostris/ai-toolkit/blob/f56b5a1d405f819c74724228564e99982624c186/toolkit/network_mixins.py) | Reuse `LoRASpecialNetwork`/`LoRAModule`, initialization, alpha/rank scaling, network activation and serialization. `module_class` is an existing customization point. Add only masked or gated residual behavior. |
| Flow scheduler | [custom_flowmatch_sampler.py](https://github.com/ostris/ai-toolkit/blob/f56b5a1d405f819c74724228564e99982624c186/toolkit/samplers/custom_flowmatch_sampler.py) | Reuse the native timestep table and noise interpolation, subject to the exact v1 sampling/reduction contract below. |

Other expected reuse points are `get_dataloader_from_datasets`, native dataset DTOs, native bucket/resize/crop and latent caches, `get_accelerator`, native progress/timer/logging helpers, metadata helpers, and safetensors/save-state utilities. Inventory the actual available interfaces before coding. New code is justified for the Gen2 mathematics, branch/phase orchestration, coherent multi-component checkpoints, and the additional diagnostics. Do not fork generic dataset, optimizer, scheduler, VAE, attention, quantization or model-loader implementations.

### 2.2 Packaging and minimal upstream changes

Keep almost all new code under `extensions/gen2_trainer/`. Suggested modules and responsibilities:

| Path | Responsibility |
|---|---|
| `extensions/gen2_trainer/__init__.py` | Lightweight registration only; do not load models during extension discovery. |
| `process.py` | Gen2 process/lifecycle bridge, using the native trainer helpers. |
| `config.py` | Typed Gen2 schema, validation, inheritance and resolved configuration. |
| `backend_ideogram4.py` | Imported Ideogram components, exact feature extraction and velocity conversion. |
| `conditioning.py` | Trigger compiler, separate embedding bank, masked adapter and region masks. |
| `gates.py` | Cubic time gates and regularizers. |
| `objectives.py` | Explicit scalar/per-example losses and their reductions. |
| `engine.py` | Phase schedule, gradient ownership, accumulation, four native optimizers/schedulers. |
| `inference.py` | Complete package loading, trigger routing, CFG branch control and sampling ablations. |
| `recording.py` / `diagnostics.py` | Structured recording, numerical probes, summaries and export. |
| `checkpointing.py` | Atomic package state, complete resume and immutable-spec linkage. |
| `tests/` | Mathematical, integration and resume acceptance tests. |
| `config/train_gen2_ideogram4.example.yaml` | Fully annotated example generated after implementation. |
| `README.md` / `IMPLEMENTATION_NOTES.md` | Usage, verified compatibility, decisions and upstream patch inventory. |

A small launcher example under `config/examples/` is acceptable. Prefer a subclass or composed bridge around native training components. Do not paste the full native trainer into the extension, and do not embed Gen2 conditionals throughout core ai-toolkit files. If the installed lifecycle lacks a necessary seam, make the smallest additive hook with unchanged default behavior, explain it, and test the stock path. Ask before a broad core refactor.

**Do not call the stock training step unchanged.** The inspected trainer assumes one optimizer; its outer loop has its own accumulation, scheduler, save/resume and OOM-skip behavior. Gen2's four optimizer families and committed-update counter must have one owner. Reuse lifecycle pieces without double stepping, double loss scaling, duplicate accelerator wrapping, unconditional re-enabling of frozen parameters, or allowing native OOM handling to advance a Gen2 phase. A short extension-owned orchestration loop is preferable to distorting the mathematical schedule to fit a single-optimizer loop.

Before implementation, produce an integration inventory naming the imported symbols, overridden methods, and any proposed original-file edits. This inventory belongs outside this immutable file.

## 3. Notation and parameter ownership

All tensors below are batched where appropriate. `stopgrad` means detachment from autograd, not copying trainable values into a different optimization target.

| Symbol / role | Meaning | Trainable when |
|---|---|---|
| `W0`, `E0`, VAE | Original DiT weights, original text encoder including vocabulary, and VAE | Never |
| `theta` / `D` | Fresh diffusion LoRA low-rank factors | Warm-up and D updates |
| `U` / `E` | Raw parameters for the learned input token vectors | A updates |
| `phi` / `T` | Masked text-encoder LoRA factors | A updates |
| `beta` / `G` | Per-block cubic gate coefficients | Calibration updates only |
| `r` | Fixed token-norm targets from initialization | Never; saved buffers |
| `q` | Canonical content caption after control-marker removal | Data |
| `C0(q)` | Frozen native text features without learned suffix | Detached constant for training |
| `C+(q;U,phi)` | Original prefix plus learned contextual suffix features | Differentiable only in A updates |
| `tau` | Toolkit noise fraction: 0 clean, 1 noise | Input, not a parameter |

Keep trainable parameter masters in float32. Use differentiable casts to the native compute dtype at module boundaries. Do not mutate master dtype as a phase transition. Keep original encoder and backbone dropout disabled; all LoRA dropout variants are zero in v1. Freeze weights with `requires_grad_(False)` rather than disabling the student computation's autograd when gradients to its inputs are required.

## 4. Trigger compilation and learned conditioning

### 4.1 Trigger is control syntax, not a vocabulary update

Use `process.trigger_word` as the single trigger source. Default example: `<gen2style>`. Match this reserved literal case-sensitively, remove all its occurrences, and trim only outer whitespace to obtain the content string. `[trigger]` may be expanded through the existing toolkit convention before this step. Do not globally normalize inner whitespace, alter requested text rendering, or rewrite content through an LLM. This marker is reserved control syntax; a request to render the literal marker as image text requires a future explicit escaping decision.

Build the styled and neutral training captions from the **same resulting q**, independent of whether a source caption contained the marker. The dataset is already associated with this run's style. Disable native mechanisms that would reinsert the trigger during the neutral branch or cache a styled representation as neutral.

Reuse native `digest_caption_string` and the native Qwen chat template. The neutral text sequence is exactly the original serialized sequence, including its generation-prompt tokens. Append M learned soft positions **after all original nonpadding tokens**, before padding. Do not insert them into a JSON string or reorder JSON fields to move the style. Do not append another original EOS/chat token after the learned positions. This placement is part of the model package.

Tokenize the complete original input without silent truncation. Reserve M positions within the configured total text-token limit. V1's overflow policy is `error`: report the sample, original length, M, and limit. Do not silently drop content, soft tokens, or closing JSON/chat syntax. No vocabulary resize is needed: obtain original token embeddings and concatenate a separate learned embedding bank via an input-embedding override. Placeholder token IDs, if an upstream API requires them, must not be used to look up the actual learned vectors or alter the vocabulary matrix.

Record original caption, marker-expanded caption, q, serialized text, original IDs, lengths, suffix mask, positions and overflow status. A training sample has one style identity; multiple dataset directories in this run are portions of the same target style.

### 4.2 Style-agnostic initialization and exact norm parameterization

Default M = 4. All styles use the same initialization recipe; it never requests a closest style description. Tokenize the fixed generic initializer string `style` without special tokens, read its frozen vocabulary rows, and average those rows to a vector a. Let r be the median L2 norm of those rows. Require nonempty IDs, r > 0 and ||a|| > 0. Define h = a / ||a||. Using the isolated initialization RNG and configured seed, draw independent standard Gaussian vectors and normalize each to a unit vector xi_i. Initialize

\[
U_i^{(0)} = r\,\frac{h+\delta\xi_i}{\|h+\delta\xi_i\|_2},\qquad r_i=r,
\]

where delta defaults to 0.01. `style` is an initialization scaffold shared by every dataset, **not** a dataset description injected into prompts. The initializer string and jitter are configurable, but no automatic per-dataset search is allowed.

At every forward, compute in float32

\[
e_i(U_i)=r_i\frac{U_i}{\max(\|U_i\|_2,\epsilon_e)},\qquad \epsilon_e=10^{-8}.
\]

Fail if a raw norm reaches the epsilon floor or becomes nonfinite; do not silently reinitialize it. There is no straight-through estimator and no in-place post-optimizer projection. The optimizer owns U; autograd differentiates through normalization. Do not project optimizer moments or claim this is a particular Riemannian optimizer. With nonzero norm, the Jacobian is `(r_i / ||U_i||) * (I - uhat_i uhat_i^T)` and supplies an independent unit-test reference. Save U, r, the original effective vectors e_init, initializer token IDs, seed and jitter. Log raw norms and effective norms separately.

### 4.3 Masked encoder adapter

Use a low-rank residual on the MLP output projection `language_model.layers[l].mlp.down_proj` of every Qwen text decoder layer. Resolve these paths against the loaded model; the reference encoder has 36 layers. Do not adapt its vision tower, vocabulary rows, attention projections, layer norms or unrelated modules in v1.

For an input vector h to an adapted linear projection:

\[
y_{li}=W^E_lh_{li}+m_i R^E_l(h_{li}),\qquad
R^E_l(h)=\frac{\alpha_E}{r_E}B^E_lA^E_lh.
\]

Here `m_i=1` only on the M learned suffix positions and zero on every original/padding position. Default rank r_E = 4 and alpha_E = 4. Use native LoRA factor initialization: A Kaiming-uniform, B zero, no trainable bias or dropout. Apply the mask to the residual **at every adapted projection**, not merely to gradients or final features. Reuse native low-rank components with a small masked subclass/wrapper. Capture the raw frozen projection output and scaled residual for the regularizer below without rerunning the entire encoder.

Qwen remains causal. The suffix may attend to all preceding original tokens and preceding suffix tokens. Original tokens cannot attend to suffix tokens. Consequently their features must match C0(q), up to measured numerical tolerance, even after U/phi change. This is an encoder-prefix invariant; it does not guarantee that the DiT preserves content in generated images.

### 4.4 Exact Ideogram feature interface and autograd

Import the native activation-layer constant and verify the reference list:

`[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]`.

These indices designate **decoder outputs**, not a library `hidden_states` tuple that may start with the input embeddings. Preserve native RoPE position construction and causal-mask construction. At each selected layer capture H[j] of shape `(B,L,d)`. The packed feature element is

\[
F[b,i,cK+j]=H[j][b,i,c],\qquad K=13.
\]

Equivalently stack `(K,B,L,d)`, permute to `(B,L,d,K)`, then flatten the last two axes. Do not use layer-first flattening `j*d+c`. Keep native original positions and assign consecutive positions to the appended suffix. Mask padding exactly as upstream. Natural-length feature lists and native `pad_text_features` are preferred over a new packing format.

The inspected `get_qwen3_vl_features` is decorated with `torch.no_grad()` and directly invokes decoder layers. Gen2 needs a differentiable sibling/adaptation in its own module, reusing original modules and mask/position utilities. Do not edit the original helper globally. Do not assume a hook on the encoder's top-level `forward` will run. Audit `get_te_has_grad`, `get_model_has_grad`, `.detach()`, native caching and any enclosing no-grad context; the actual positive conditioning-to-loss path must remain connected during A updates.

Frozen DiT weights still participate in backpropagation to C+ on A updates, and the DiT forward must also retain the path to beta on G updates even though the ordinary LoRA factors are frozen. Use gradient-safe native checkpointing; any additional encoder checkpointing must be non-reentrant and preserve the suffix mask and branch context during recomputation. For v1, disable all stock full-text-feature/sample-prompt caching and encoder unloading. Latent caching remains allowed. Do not add a second custom generic cache implementation. A future prefix/KV cache is an optimization requiring separate parity evidence.

## 5. Diffusion LoRA and temporal gates

### 5.1 Exact default target set

Use the native LoRA implementation, with `linear=32`, `linear_alpha=32`, no bias, dropout, DoRA/LoKr variants or merging during training. Adapt exactly these five linear projections in every actual DiT block b:

`layers.b.attention.qkv`, `layers.b.attention.o`, `layers.b.feed_forward.w1`, `layers.b.feed_forward.w2`, `layers.b.feed_forward.w3`.

This is 170 adapted projections for 34 blocks. Keep input projections, `llm_cond_proj`, timestep embeddings, AdaLN projections, final projection and norms frozen and unadapted. Rank and alpha are configurable; the v1 target set is fixed. Produce the resolved path/shape/rank/alpha manifest before training and fail on missing, extra or duplicate targets. Do not infer the block from an unverified substring in serialized LoRA keys; map original module paths to explicit integer block IDs.

For projection j in block b:

\[
y_{bj}=W^D_{bj}h+\ell\,s_D\,g_b(\tau)R^D_{bj}(h),\qquad
R^D_{bj}(h)=\frac{\alpha_D}{r_D}B^D_{bj}A^D_{bj}h.
\]

`ell` is the explicit branch-level LoRA-enable flag, independent of trigger presence. It is one for both training students and zero for the base teacher. `s_D=1` during training; an inference strength is separately configurable. The gate multiplies only the LoRA residual. It must never scale the frozen projection, an entire transformer block output, native AdaLN gates, or the full velocity prediction.

### 5.2 Exact gate function

For each of the 34 blocks, learn four coefficients beta_b0...beta_b3, initialized to zero. Define

\[
\psi_k(\tau)=\binom{3}{k}\tau^k(1-\tau)^{3-k},\quad
p_b(\tau)=\sum_{k=0}^3\beta_{bk}\psi_k(\tau),\quad
g_b(\tau)=1+\rho\tanh p_b(\tau).
\]

Default rho = 0.5; require 0 < rho < 1. Thus `1-rho < g < 1+rho`, zero beta gives g=1, and the parameter count is 136. Rho and the regularizer strengths are engineering controls, not proven optima. During warm-up and alternating refinement, use exactly g=1 and keep beta frozen. During calibration, train beta and freeze theta/U/phi.

One gate is shared across the five adapted projections and all token regions of its block. Different examples in a batch may have different tau; broadcast the gate along sequence/feature axes without averaging over the batch. Use the toolkit noise fraction, **not** the internal reversed Ideogram time. No content-to-gate network, rank masking, negative gate, per-token gate or independent gate for every projection is part of v1.

Do not pass a trainable gate through a native setter that converts it to a Python float, calls `.item()`, constructs a detached tensor, or averages batch multipliers. Keep native network strength at the fixed branch value and apply the differentiable gate to the native residual through a scoped subclass/wrapper. Native factor scaling must appear exactly once.

## 6. Training states, branches and losses

### 6.1 State sampling and time convention

Use native image loading, bucket transforms, VAE encoding and latent normalization. Let z0 be the resulting clean model latent and epsilon independent standard Gaussian noise from the native noise utility with all offset/correction/multiplier modifications disabled. The reference latent layout is the native patchified `(B,128,gh,gw)` representation.

Use `CustomFlowMatchEulerDiscreteScheduler.set_train_timesteps` with `timestep_type="linear"` and N=`train.num_train_timesteps` (default 1000). It supplies the table `linspace(1000,1,N)`. Draw each example's index uniformly from **all** indices 0 through N-1. The v1 index upper bound is inclusive through N-1; do not inherit an off-by-one endpoint omission from a generic batch helper. Let `timestep = table[index]` and `tau = timestep/1000`. Reuse native interpolation and verify

\[
z_\tau=(1-\tau)z_0+\tau\epsilon,\qquad v^*=\epsilon-z_0.
\]

This is a finite uniform time-table distribution, not a claim of continuous-uniform sampling. No additional SNR weighting, time-density correction, resolution shift, signal correction, noise offset or prediction rescaling applies in v1. Changing timestep distribution is a later explicit experiment, not an undocumented default.

The native `predict_velocity` already passes `1-tau` to Ideogram and negates its native clean-minus-noise prediction. Reuse it and **do not negate twice**. The time gates use tau outside that conversion. Training and teacher predictions are conditional velocity predictions without CFG, even though image sampling later uses CFG.

### 6.2 Three distinct forwards

For each paired example, use exactly the same q, z0, epsilon, tau, latent transform and model compute settings:

| Name | Features | Diffusion LoRA | Gates | Target / autograd |
|---|---|---|---|---|
| `base_teacher` | C0(q), no suffix | Disabled | Bypassed | Detached native base velocity v0 |
| `styled_student` | C+(q;U,phi) | **Enabled** | One or learned, according to stage | Dataset target v*; gradient to the active family |
| `neutral_student` | C0(q), no suffix | **Enabled** | The same stage's gate function | Detached base-teacher prediction; gradient to D or G |

The neutral student is not the CFG unconditional branch. It still receives the content prompt. A teacher with a LoRA accidentally active, or a neutral student with it disabled, invalidates the experiment.

One loaded frozen backbone is sufficient: temporarily disable the new diffusion network to compute the teacher under `no_grad`, then restore state in `try/finally`. Original frozen weights must not change. The teacher uses the conditional base weights, not an EMA student or the separate optional unconditional adapter. A second full model copy is unnecessary. Native `get_prior_prediction` may be reused only after verifying its conditioning, adapter disable/restore and no-CFG semantics match this table.

The neutral states in v1 are re-noised target-dataset latents, paired with their captions. This controls compute and makes comparisons exact. It does not cover all states on base-model sampling trajectories. Document this scope; do not claim global neutral preservation from this training loss. Replay trajectories or a synthetic data curriculum are not part of v1.

### 6.3 Exact scalar reductions

For example i, let mean_x average over every valid image-latent element, excluding any padding if a backend introduces it. Reduce examples with an arithmetic mean, so examples with more spatial elements do not automatically receive greater weight:

\[
L_{S,i}=\operatorname{mean}_x (v^+_i-v_i^*)^2,\quad
L_{N,i}=\operatorname{mean}_x (v^-_i-\operatorname{sg}(v_{0,i}))^2,
\qquad L_S=\frac1B\sum_i L_{S,i},\quad L_N=\frac1B\sum_i L_{N,i}.
\]

Compute residual differences, squares and scalar reductions in float32 even when model forwards use bf16. L_N is always nonnegative; do not negate it or replace it with divergence maximization. In both branches the weights above are one. A native loss helper may be reused only if its exact target, weighting, normalization and auxiliary terms are equivalent.

### 6.4 Text-adapter regularizer

At encoder projection l and suffix position k for example i, let b_ilk be the original frozen linear output `W^E_l h`, and a_ilk the alpha/rank-scaled LoRA residual R^E before addition. Define

\[
R_T=\operatorname{mean}_{i,l,k\in\mathrm{suffix}}
\frac{\operatorname{mean}_c a_{ilkc}^{2}}
{\operatorname{sg}(\operatorname{mean}_c b_{ilkc}^{2})+10^{-6}}.
\]

Average equally over examples, adapted encoder layers and the M suffix positions. The denominator is detached so its growth cannot reduce the penalty by that gradient path. Do not detach the numerator. This is an output-size regularizer, not a style metric or a guarantee against all hidden-state drift. Log numerator and denominator distributions as well as the ratio. There is no embedding-cosine loss or semantic-description target.

### 6.5 Gate regularizers

On a fixed grid of Q=65 equally spaced points including 0 and 1, use

\[
R_C=\frac1{34Q}\sum_{b,j}(g_b(\tau_j)-1)^2,\qquad
R_H=\frac1{34Q}\sum_{b,j}(g'_b(\tau_j))^2.
\]

The exact derivative is `g' = rho * (1-tanh(p)^2) * p'`, with

\[
p'_b(\tau)=3\{(\beta_{b1}-\beta_{b0})(1-\tau)^2+
2(\beta_{b2}-\beta_{b1})\tau(1-\tau)+
(\beta_{b3}-\beta_{b2})\tau^2\}.
\]

Compute these terms differentiably in float32 using the fixed grid, independently of the minibatch's sampled times. Q is configurable; Q >= 5. The mean over blocks and grid points is mandatory so changing Q does not multiply the effective weight.

When encoder layers are checkpointed, return each layer's differentiable, per-example R_T contribution as an explicit output of its checkpointed wrapper and aggregate it once. Do not accumulate a differentiable regularizer through an uncontrolled forward-hook list: recomputation can duplicate entries or retain the entire activation graph. Diagnostic hooks must suppress recomputation duplicates or mark and exclude them from normal aggregates. This bookkeeping requirement applies equally to additional gradient-probe forwards.

### 6.6 Stage-specific objectives and gradients

| Update kind | Objective | Parameters receiving optimizer updates |
|---|---|---|
| Warm-up D | `L_S + lambda_N * L_N` with g=1 | theta only |
| Refinement D | `L_S + lambda_N * L_N` with g=1 | theta only |
| Refinement A | `L_S + lambda_T * R_T` with g=1 | U and phi only |
| Calibration G | `L_S + lambda_N * L_N + lambda_C * R_C + lambda_H * R_H` | beta only |

Pilot defaults: lambda_N=1.0, lambda_T=1e-4, lambda_C=1e-3, lambda_H=1e-4. These are explicit starting settings, not empirically established values. All four weights are configurable and nonnegative; zero is a labeled ablation. Lambda_N=0 removes the learned-neutral training constraint and must be prominently recorded.

Do not compute L_N for optimization on A updates: the neutral branch has no U/phi and cannot train them. It may be evaluated separately as a diagnostic. Conversely, keep L_N active on gate updates, since gate changes could damage neutral preservation. R_T is not added to D/G updates, and gate penalties are not added before calibration. Do not introduce extra auxiliary losses without asking.

## 7. Schedule, optimizers, accumulation and execution

### 7.1 One authoritative update counter

`logical_update` counts committed updates, not microbatches, forward passes, scheduler calls or individual optimizer calls. Start at zero before training. Default schedule:

| Stage | Logical updates | Pattern |
|---|---:|---|
| Warm-up | 450 | D only |
| Refinement | 2100 | Repeat D,D,D,D,A |
| Calibration | 450 | G only |

The total is `train.steps=3000`. The resulting successful optimizer-step counts are D=2130, E=420, T=420, G=450. An A update steps E and T once each but advances `logical_update` by only one. The stage allocations and positive integer D/A cycle lengths are configurable. When refinement ends partway through a cycle, use the deterministic prefix of that cycle and record the resulting optimizer-local totals before training. Zero-length stages are allowed as explicitly recorded ablations; the stage lengths must sum exactly to `train.steps`.

Save a coherent initialization checkpoint at update zero, every configured save interval, and at every stage boundary and the end. Phase boundaries are determined by committed counters only. All sample, probe and logging intervals in this specification use committed logical updates unless explicitly labeled microbatch or optimizer-local.

When an interval and a boundary coincide, perform each identical save/probe once. For image sampling take the union of due mode/seed/prompt sets and deduplicate identical requests by checkpoint hash plus full sampling settings. Record all reasons that triggered the event.

### 7.2 Four native optimizer families

Use four independently owned native optimizer instances: diffusion D, embeddings E, encoder adapter T and gates G. E and T optimize the same A objective with different configurable learning rates. This avoids requiring every supported optimizer to implement special unequal parameter-group semantics. The four parameter sets must be disjoint, and their union must equal all trainable Gen2 parameters.

Inherit optimizer name, kwargs and scheduler settings from native `train` settings unless overridden in `gen2.optimizers.<role>`. Call the existing factories for each role. Default AdamW-family rates are D=1e-4, E=1e-3, T=1e-5 and G=1e-3; default weight decay is zero. These rates are pilot settings and do not claim applicability to every optimizer.

**All upstream-supported optimizers remain selectable.** Forward names/kwargs to the installed factory; do not translate everything to AdamW or implement a Gen2 optimizer registry. Preserve native special behavior, such as learning-rate handling for Prodigy/DAdaptation, and log requested and actual rates. Do not inject extra kwargs such as `eps` if the upstream factory already supplies them. Report missing optional dependencies and invalid factory arguments clearly, with no silent fallback. Use existing optimizer-specific train/eval or offload synchronization hooks when applicable; support is not limited to optimizers that happened to be tested on the implementation machine.

Use native LR-scheduler construction with **optimizer-local horizons** computed from the schedule. Advance a scheduler only when its optimizer actually commits an update. Do not restart D's schedule at the warm-up/refinement boundary or tick E/T during D updates. Preserve explicit valid scheduler kwargs, adapting factory conventions rather than implementing a new scheduler. Record effective constructor arguments. Optimizers with their own LR adaptation retain it; report both externally requested LR and any native effective-LR statistic.

For gradient clipping, reuse the installed toolkit's optimizer-specific policy with the configured `train.max_grad_norm`. The inspected policy applies norm clipping except for Adafactor. Apply the appropriate policy to each active family, log whether it was applied, and record pre/post norms. Inactive optimizers are neither stepped nor allowed to decay their parameters. Clear their `.grad` to None when changing ownership. Never delete optimizer state on a phase switch.

### 7.3 Accumulation and mixed precision

An update consumes `gradient_accumulation_steps` microbatches of the same update kind; do not switch D/A/G inside an accumulation window. Negative or epoch-sized accumulation is unsupported in v1. Let microbatch j contain n_j examples and N=sum_j n_j. Its data and R_T loss contribution is weighted by n_j/N, so unequal final microbatches do not distort the mean. The gate-only grid regularizers are added **once per logical G update**, not once unscaled per microbatch.

Use native accelerator/backward and precision utilities, but have a single loss-scaling owner. Verify that accelerator accumulation does not divide by the accumulation factor a second time. Default compute is bf16; if fp16 scaling is selected, unscale all active gradients before finiteness checks and clipping. On A updates, E and T commit together only after all active gradients are acceptable. A skipped/invalid update must not tick any optimizer, scheduler, phase or committed counter.

Default training-error policy is `abort`: flush a failure record and preserve the last complete checkpoint. Do not inherit native silent OOM batch skipping that advances the schedule. An exception after one of two A optimizers has stepped is a partially committed failure; record it and do not label or save those weights as a valid completed update. Resume from the last coherent checkpoint. Automatic numerical recovery, reduced resolution, changed batch size or altered losses require a user decision.

### 7.4 Required order and context safety

For a D or G update, the following sequential structure is an admissible reference. Equivalent accumulation may be used only if the same gradient is obtained:

```python
kind = schedule.next_kind(committed_update)
set_exact_gradient_ownership(kind)
zero_active_gradients()
for batch, weight in accumulation_window:
    # Reuse native data/latent/noise preparation, honoring Section 6.1.
    q, z0, noise, tau = prepare_paired_batch(batch)
    C0 = encode_native_content_no_grad(q)
    z_tau = add_native_noise(z0, noise, tau * 1000)
    if kind in ("D", "G") and lambda_N > 0:
        with base_teacher_context():
            v0 = predict_native_velocity(z_tau, tau, C0).detach()
    with styled_student_context(kind):
        Cplus = encode_gen2(q)  # enable autograd only for A
        LS, RT = styled_objective(...)
        backward(weight * (LS + (lambda_T * RT if kind == "A" else 0)))
        # Finish backward/recomputation before leaving mutable branch context.
    if kind in ("D", "G") and lambda_N > 0:
        with neutral_student_context(kind):  # LoRA MUST remain enabled
            LN = neutral_objective(..., teacher=v0)
            backward(weight * lambda_N * LN)
if kind == "G":
    RC, RH = gate_regularizers_on_fixed_grid()
    backward(lambda_C * RC + lambda_H * RH)  # once, fixed-grid objective
check_finite_and_clip_active_gradients()
step_only_active_native_optimizers_and_schedulers()
commit_logical_update_and_record()
```

In an A update, skip teacher/neutral forwards for optimization even when lambda_N is nonzero. Warm-up uses kind D and initialized fixed conditioning. Computing detached C+ on D/G updates is valid; computing detached C+ on A updates is invalid.

Do not change adapter-active flags, suffix masks, gate context, device placement or train/eval modes while an outstanding checkpointed graph may recompute with those values. Bind the relevant tensors to the recomputation closure or finish backward inside the same scoped context. Use `try/finally` for every state change. Global monkey patches or shared mutable flags without a defined lifetime are not acceptable.

Disable parameter swapping, graph compilation, stock embedding training, stock encoder training, extra regularization datasets, extra adapters, model merging and EMA in the default v1 profile. The custom encoder adapters are trained despite native `train_text_encoder: false`; that native flag must never unfreeze the original encoder. For options incompatible with these invariants, raise a configuration error rather than silently honoring part of the request. Native quantization and offload are allowed only when tested to preserve A-step input gradients. Do not implement new quantizers or offload engines.

## 8. Inference and faithful visual comparisons

Provide a loader and sampling entry point for the complete Gen2 package, reusing native model loading, latent initialization, Euler schedule, packing, VAE decode and image-save machinery. A bare LoRA export is incomplete; do not present it as equivalent.

When the trigger is present, remove the control marker and encode C+. By default, when it is absent, encode C0 **with the new diffusion LoRA still enabled** (`missing_trigger_policy: learned_neutral`). This allows the learned-neutral behavior to be observed. An optional `base_bypass` policy may turn off the package for a hard neutral switch, but that behavior is not evidence of learned trigger dependence. Diagnostic neutral generation must always force the LoRA-enabled route regardless of production policy.

New personalization LoRA residuals and time gates apply only to the conditional CFG pass. The CFG unconditional pass has no text tokens and the new LoRA is disabled. If the user configures the existing native unconditional LoRA, retain it as frozen and enable it only on that unconditional pass. It must never enter the neutral teacher or either training student. Record its identity or absence.

For sampled noise fraction tau, compute gates using tau; call native velocity conversion; reuse

\[
v_{\mathrm{cfg}}=v_u+\gamma(v_c-v_u),\qquad
z_{\mathrm{next}}=z+v_{\mathrm{cfg}}(\tau_{\mathrm{next}}-\tau).
\]

For guidance <=1 follow native no-CFG behavior. The inspected native preview loop toggles the optional unconditional adapter but does not itself guarantee that the main LoRA is off during the unconditional call. Gen2 must enforce its explicit branch contract rather than assume that stock behavior matches it.

Use the same conditioning implementation and gate function in training, previews and exported inference. Do not train four tokens and sample one literal trigger token. Disable stock sample-prompt caching of trainable features. Regenerate them from the current checkpoint. Preserve training modes, optimizer modes, phase masks and all training RNG streams across evaluation.

### 8.1 Required ablation modes

All comparisons use identical content prompts, initial noise, sampler, guidance, resolution and checkpoint component versions. Each mode must be explicitly recorded:

| Mode ID | Intervention |
|---|---|
| `full` | Learned U/phi, diffusion LoRA enabled, learned stage-appropriate gates. |
| `neutral_lora_on` | No suffix; diffusion LoRA and stage-appropriate gates enabled. |
| `base` | No suffix; new diffusion LoRA disabled. |
| `base_with_conditioning` | Learned U/phi and suffix, new diffusion LoRA disabled. This is not an unpersonalized encoder condition. |
| `conditioning_init` | Same M-position layout, restore effective e_init and disable encoder adapter; keep current diffusion LoRA/gates. |
| `encoder_adapter_off` | Learned U, encoder adapter disabled; current diffusion LoRA/gates. |
| `tokens_init` | Restore e_init while retaining learned encoder adapter; current diffusion LoRA/gates. |
| `gates_one` | All g=1; all other learned components retained. |
| `gates_time_mean` | For each block use the arithmetic mean of its g over the fixed gate-regularization grid, constant throughout sampling. |

Use isolated parameter views/contexts, not destructive overwrites of live training tensors or optimizer states. `conditioning_init` preserves the token count and is needed to distinguish a learned representation's contribution from the mere presence of added positions. Ablating a co-adapted component is a causal intervention on that checkpoint, not a matched-budget retraining experiment.

Provide a small frequent preview panel and a larger milestone panel. Prompts are tagged `easy`, `hard` or `unseen` by the user; do not invent empirical classifications. Save individual images with JSON metadata, then optional labeled contact sheets. The user judges images; numerical diagnostics must not claim a visual score unless an actual external evaluation provides it.

## 9. Comprehensive recording contract

Recording is a first-class part of the implementation. It must support analysis without access to the running Python process and without requiring the original training machine. Use machine-readable UTF-8 JSON/JSONL plus optional CSV exports. Do not depend solely on console output, TensorBoard or a remote service. Native loggers may receive the same metrics additionally.

### 9.1 Run directory and provenance

Use the native training output folder with a `gen2/` child. Minimum outputs:

| Artifact | Required contents |
|---|---|
| `run_manifest.json` | Run UUID, spec version/hash, actual ai-toolkit/code revisions, dirty diff hash, package/Python/PyTorch/CUDA/driver versions, GPU memory/model, precision per component, attention backend, quantization/offload settings, model/tokenizer/VAE revisions and weight identities, seeds, time convention, phase totals, optimizer-local horizons and resolved target counts. Never dump authentication tokens or arbitrary environment variables. |
| `config.requested.yaml`, `config.resolved.yaml` | Original config and fully expanded effective defaults/inheritance. Include which settings are fixed by v1, selected ablations, and effective native factory arguments. Secrets are excluded. |
| `dataset_manifest.jsonl` | Stable sample ID, relative path, content hash, original/canonical caption and hashes, split, optional content-group label, source dimensions, caption provenance and duplicate/leakage status. |
| `module_manifest.json` | Every adapted/frozen target identity, original path, tensor shape, parameter family, rank, alpha, block ID, dtype/device and parameter count. |
| `events.jsonl` | Initialization, phase transitions, save/reload, evaluations, warnings, exceptions, aborted/partial updates, configuration validation and capability checks. |
| `microbatches.jsonl` | Example-level state/target/loss summaries and branch provenance for every training microbatch. |
| `updates.jsonl` | Every committed logical update's aggregate losses, active families, optimizer counters/rates, gradient and update summaries, timing and memory. |
| `activations.jsonl` | Scheduled per-layer/per-region residual summaries. |
| `probes.jsonl`, `gates.jsonl`, `spectra.jsonl` | Fixed numerical probes, gate curves, gradient decomposition and low-rank/embedding diagnostics. |
| `samples/manifest.jsonl` | Every generated image's complete reproducibility and ablation metadata. |
| `human_ratings.csv` | An initially unscored, appendable template joined by image/prompt/checkpoint IDs. |
| `summary.json`, `summary.md` | Reproducible aggregates with explicit missing-data counts, stage boundaries and descriptive findings; no invented visual conclusions. |

Every event/metric row includes schema version, run ID, logical update, update-attempt ID, stage, update kind and timestamp. Microbatch rows also include accumulation index and example IDs. Use null plus a reason for undefined metrics; do not use zero to mean missing. Store only finite JSON numbers; nonfinite model results create explicit error records. Metrics must distinguish raw vs weighted, pre-update vs post-update, exact vs sampled, and training vs diagnostic passes.

### 9.2 Every microbatch and update

For **every example**, record the selected time-table index, native timestep, tau and internal Ideogram time, bucket/crop/flip parameters where native metadata exposes them, latent dimensions, valid-element count, z0/noise/z_tau/target RMS, source sample ID, token lengths and branch IDs. Record exact per-example L_S and, on D/G updates when enabled, L_N. Include caption variants or augmentation choices actually used. Do not fabricate a replay seed when an upstream transform did not expose one; record the available RNG state identifier and mark that limitation.

For **every logical update**, record:

1. Microbatch/example counts, effective batch size, all optimizer-local counters and phase/cycle position.
2. L_S, L_N, R_T, R_C and R_H separately, their coefficients, weighted contributions and the active total. Inapplicable terms are null with `not_in_objective`, not falsely reported as zero observations.
3. Requested/actual LR per active family and native adaptive-LR statistics when available; scheduler state/counter, weight decay, optimizer identity and clipping policy.
4. Full-family gradient L2 norm, RMS, maximum absolute value, finite fraction and nonzero fraction before clipping; post-clip norm; counts of missing-gradient tensors. Respect legitimate zero-initialized-factor behavior.
5. Parameter norms and sampled update magnitudes using deterministic stored coordinate indices. Log sample size/coverage and mark these estimates `sampled`. At configured intervals optionally compute full update norms with bounded temporary memory; never label a coordinate estimate exact.
6. Encoder/teacher/styled/neutral/backward/optimizer/diagnostic/I/O timing, total update time, peak allocated/reserved GPU memory and host RSS when available. CUDA synchronization used for timing must be documented; approximate timings must be labeled.
7. Current active-state assertions, cache/feature version IDs, loss scaling, skipped/failed attempt counts, and recorder queue/disk budget status.

The default update-coordinate budget is 65,536 deterministic coordinates per parameter family, sampled uniformly without replacement from that family's flattened coordinates; inspect all coordinates for smaller families. Store the index selection once. An estimated squared L2 norm is the sampled squared sum multiplied by total_coordinates/sample_coordinates. All original weights must stay frozen; verify structural ownership each update, exclude original weights from every optimizer, and verify their identity in acceptance checks. Compare full trainable-family state hashes at phase boundaries and acceptance checks. Do not silently hash a tiny sample and call that full frozen-weight verification.

### 9.3 Activation and representation diagnostics

At `activation_every` updates, collect detached summaries for every adapted diffusion projection and every adapted encoder projection. Label branch, original module path, block/layer ID, tau bin and token region. Separate DiT regions into original text, learned suffix, image and padding using actual packing masks. Neutral branches have no suffix; do not synthesize one for logging.

Minimum summaries are input RMS, original frozen linear-output RMS, un-gated LoRA-residual RMS, applied residual RMS, residual/base ratio with denominator recorded, and residual/base cosine where defined. For the encoder include masked-write leakage outside the suffix, regularizer numerator/denominator, and prefix differences relative to a frozen native pass. Reduce on-device and transfer scalars; do not retain every full activation graph.

For each learned token, record raw/effective norm, cosine to its saved initialization, pairwise token cosine matrix and effective-vector singular values. For the suffix record RMS by encoder tap and examples. These expose norm drift, collapse and context sensitivity. Large attention weight or residual size is not a percentage of learned knowledge retrieved; do not invent an activation-utilization percentage.

At scheduled checkpoints, optionally compute low-rank spectra without forming a dense BA matrix: thin-QR B and A-transpose, then take singular values of the small product R_B R_A-transpose, including alpha/rank scaling. Report Frobenius norm, largest singular values, and the rank needed to explain 95% of **squared-singular-value energy**. Record the convention. Use the same technique for encoder factors where useful. For effective-rank entropy define p_j=sigma_j^2/sum(sigma^2); the all-zero update has undefined entropy rank, recorded explicitly.

### 9.4 Fixed numerical probe suite

Create a probe manifest before training. Select fixed real dataset/validation examples, freeze their preprocessed z0, draw fixed epsilon with the diagnostic RNG, and use a configured list of tau values spanning low/mid/high noise. Save the small latent/noise packet with IDs and hashes. Reuse it at every checkpoint; probes must not perturb training RNG, minibatch order or parameters. Do not use training probe losses to label an unseen-content visual prompt as measured reconstruction generalization when no target image exists.

On each probe state, compute this 2x2 intervention table using the current conditioning and gates:

| Prediction | Diffusion LoRA | Conditioning |
|---|---|---|
| v11 | On | Learned suffix |
| v10 | On | Neutral C0 |
| v01 | Off | Learned suffix |
| v00 | Off | Neutral C0 |

Record RMS values of `v11-v10`, `v01-v00`, `v10-v00`, `v11-v01`, and interaction

\[
I=(v_{11}-v_{10})-(v_{01}-v_{00}).
\]

Also record the relevant pairwise cosines, reference RMS denominators, per-example reconstruction losses where real targets exist, and conditioning-prefix differences. These quantify response and interaction. They are not style scores, and interaction magnitude alone does not establish a helpful direction.

Compare v00 on identical fixed states against its initialization reference, recording frozen-base prediction drift and numerical tolerance. Its inputs and original weights are supposed to stay fixed. An unexplained drift is a backend/context/frozen-state issue to investigate, not style learning. Store sufficient reference scalars or bounded reference predictions to perform this comparison.

At `gradient_probe_every`, measure gradients of L_S and lambda_N L_N separately with respect to theta on the fixed probe, without optimizer updates. On conditioning probes separate L_S and lambda_T R_T gradients for U and phi. Report per-family norms, norm ratios and cosine conflict. Use `autograd.grad` or an equivalent isolated mechanism that preserves training `.grad` and optimizer state. Exact full-family gradients are preferred within the memory budget; if a deterministic coordinate subset is necessary, explicitly label sampled cosine/norms and coverage. Do not retain two full model graphs or change context before checkpoint recomputation completes.

### 9.5 Gate diagnostics

Record beta and g for every block on the same fixed regularization grid at initialization, every configured gate interval and every calibration checkpoint. Include per-block mean/min/max, derivative RMS, R_C/R_H, and saturation fraction where `abs(tanh(p)) >= 0.95`. Record the actual sampled-time distribution separately. This distinguishes a learned time schedule from a constant block rescaling or bound saturation. `gates_time_mean` uses this grid's arithmetic mean, not a newly optimized constant.

### 9.6 Visual metadata and human evaluation

Each image record must include checkpoint/package hash, prompt ID/group/text/actual serialization, trigger route, ablation mode, component identities, initial noise seed and RNG/backend provenance, sampler/sigma schedule, sample steps, resolution, CFG scale, LoRA strength, unconditional-adapter identity, gate mode and inference code revision. Keep image filenames stable and collision-free.

Human-rating columns must include image ID, rater, date, target-style fidelity, content fulfillment, visible artifacts, copying/suspicious similarity, and notes. Leave score fields empty. Ratings for generic Ghibli resemblance are not automatically ratings for the actual target dataset. Automated style/content metrics are optional later evaluation integrations; they are not a mandatory v1 training loss and must never silently replace the user's visual assessment.

The summary utility must aggregate by stage, checkpoint, time bin, content group and split; show distributions, counts, standard deviations and lower-performing groups where actual scores exist. Use paired differences for matched seeds/prompts. Do not pool incomparable raw D/A/G objectives into a single progress claim. Distinguish validation used for iteration from final test prompts.

### 9.7 Recorder reliability and size limits

Use append-only JSONL with a bounded writer queue, configurable flush interval and size-based rotation. Apply backpressure rather than silently dropping core rows. Never move live autograd graphs to the recorder. Flush at checkpoints and on orderly errors. On resume, retain previous rows, log a resume event and use attempt IDs to distinguish any replayed work.

Raw tensor dumps are optional, disabled by default, and bounded by configured packet count and byte budget. Core scalar/provenance records must remain enabled. If a core-recording budget or disk limit is reached, stop cleanly with an explicit error; do not run silently without diagnostics. Optional oversized tensor dumps may be skipped with an explicit reason. Provide a diagnostic export command that bundles manifests, configs, JSONL/CSV summaries, fixed probe packet and sample metadata. Include images only when requested; exclude model weights and the original dataset by default.

## 10. Checkpoint and resume contract

Use native serialization helpers for supported components, with a Gen2 package manifest. Save at minimum:

- Native diffusion-LoRA weights and their target/alpha/rank mapping.
- U, fixed r, e_init and initializer metadata; masked encoder-LoRA weights/mapping.
- Beta, rho, basis identity, gate grid convention and feature-tap/packing contract.
- Four optimizer states, four scheduler states, scaler if used, family step counts, committed update, stage/cycle position and accumulation status.
- Training/diagnostic/preview RNG states, sampler/dataloader resume information available from native mechanisms, and recorder offsets/last event IDs.
- Requested/resolved configuration, all model/tokenizer/VAE identities, spec hash/version, code revision and complete component hashes.

Save only at accumulation boundaries for v1. Use a temporary checkpoint directory, finish all component writes and hashes, then publish the manifest/complete marker atomically. A directory without its complete marker is not resumable. Coordinate native retention so stage-boundary checkpoints and their diagnostics remain available even when rolling checkpoints are pruned. Do not save only the currently active family, only float16 inference weights, or only the base trainer's optimizer state and call it resumable. Inference exports may use the configured save dtype, but resume must retain float32 trainable masters and optimizer precision.

On resume, verify spec hash, model identities, module mapping, token layout, phase schedule and all component checksums. Strict resume rejects missing or mismatched states. Loading weights with reset optimizer/RNG state is a distinct `weights_only_restart` operation with a new run ID, not resume. Require a user decision for incompatible resume settings.

Exact continuation is verified under the same backend/environment and deterministic data loading. If a native multi-worker loader cannot restore prefetched augmentation state, explicitly record that limitation and provide the reference deterministic mode with zero loader workers. Do not claim bitwise replay across hardware, quantization, CUDA versions or undocumented loader state.

## 11. Acceptance tests and implementation completion gates

Tests here verify real risks in the new mathematics and integration. They are required even though a long training run is not a unit test.

1. **Registration/reuse:** `job: extension` with process `gen2_trainer` discovers the new process without changing the behavior of stock `sd_trainer`. Verify optimizer/scheduler factory calls and effective kwargs for all four families; do not test a reimplemented optimizer.
2. **Embedding math:** float64 toy tests compare the normalized-token Jacobian/VJP with the analytic formula and a finite difference. Vocabulary weights stay unchanged; e norms equal fixed r. Save/reload preserves e_init and current e.
3. **Causal mask:** a small causal encoder fixture with nonzero adapters proves arbitrary suffix perturbations cannot change original positions. A deliberately wrong unmasked implementation must fail the test. On the real backend compare native prefix features at all 13 taps within documented dtype tolerances.
4. **Packing:** uniquely tagged synthetic tap/channel values prove `c*K+j` ordering, layer-output indexing, valid masks and suffix positions. A no-suffix, adapters-off pass matches upstream encoding. Vary prompt lengths and include the token-limit error path.
5. **Flow convention:** an analytic interpolation fixture checks both endpoints, target sign, time scaling and a one-step Euler update. The gate at toolkit tau must not receive internal `1-tau`. Verify native helper parity rather than adding a second conversion.
6. **LoRA and gates:** at g=1 match native LoRA output; at ell=0 match native base output. Check alpha/rank once, batch-dependent gates, the five-projection/block mapping, all-zero beta and the analytic gate derivative/regularizers. No dense-base output is scaled.
7. **Teacher/student roles:** with fresh zero LoRA output, neutral prediction matches teacher. With deliberately nonzero LoRA factors, neutral student keeps a nontrivial path to D. Teacher receives no parameter gradients, uses C0 and restores state after exceptions. The CFG unconditional pass is tested separately.
8. **Gradient ownership:** on controlled nonzero-gradient fixtures, a D update changes only theta, an A update changes only U/phi, and a G update changes only beta. Inactive optimizer states/counters do not advance. Zero B can legitimately make A-factor gradients initially zero; do not write an assertion demanding every parameter have nonzero gradients at step one.
9. **Frozen-backbone input gradients:** production A forward retains derivatives to U/phi with the selected checkpointing, dtype and quantization settings. Verify actual consumed-feature changes, a real parameter update and reloaded inference changes. A mocked encoder or optimizer is insufficient for this integration check.
10. **Accumulation:** two unequal-size microbatches produce the same gradient as their concatenation in a deterministic fixture. A grid regularizer is applied once. Test schedule boundaries, partial cycles, local LR counters and A's two optimizer steps.
11. **Checkpoint/recompute contexts:** checkpointing on/off yields compatible gradients in a controlled fixture; alternating styled/neutral/teacher contexts does not leak flags, masks or times. Test exception restoration and no mutation during outstanding recomputation.
12. **Resume:** continuous N+M updates versus N, save/reload, M updates must agree within the reference environment's documented tolerances, including phase/optimizer counters and diagnostic IDs. Missing component files or a modified spec hash must fail strict resume.
13. **Inference parity:** identical package state and fixed seeds agree between preview and exported loader. Ablations are deterministic, preserve token count where specified, and restore the live training package. Validate styled and neutral CFG paths.
14. **Recorder:** validate schemas, joins, finite/null handling, resume append behavior, bounded queue, rotation, budget errors and diagnostic export. Diagnostic and preview RNG must not change a subsequent training draw. Unscored human-rating cells remain empty.
15. **Annotated config:** parse the final YAML, validate every new leaf key against the implemented schema, reject misspellings/conflicts, and show a fully resolved dry run without loading weights. A short real backend smoke run covers D, A and G before claiming implementation complete; report the hardware and any test limitations.

Do not loosen tolerances or silently disable checkpointing/quantization to turn a failing production test into a pass. Compare measured numerical noise with the configured tolerance, explain the failure, and ask when a consequential adjustment is needed. Optional dependencies absent from the test machine must be labeled untested, not claimed unsupported by design or falsely tested.

Deliver the extension code, test results, integration/patch inventory, usage instructions, diagnostic export command and **a fully commented example YAML generated after implementation**. Keep this specification unchanged. Do not launch a long production training run merely to conceal unresolved correctness failures; implement the short acceptance run as part of normal validation and ask about missing dataset/hardware choices before a substantive run.

The ordinary launch must work through the existing entry point, for example `python run.py extensions/gen2_trainer/config/train_gen2_ideogram4.example.yaml`. Provide documented extension commands for configuration-only validation, complete-package inference, and diagnostics export. Their exact CLI spelling may follow repository conventions; do not require a replacement ai-toolkit launcher.

## 12. Configuration contract and fully annotated example

### 12.1 Schema requirements

The final implemented example must keep the familiar ai-toolkit shape: `job: extension`, `config.name`, `config.process`, `type: gen2_trainer`, and native `network`, `datasets`, `train`, `model`, `save`, `sample` sections. Put Gen2-specific options under `gen2`. Do not create a new configuration language or require editing Python constants for normal experiments.

The block below is a **syntactically valid configuration contract**, with path placeholders. It becomes a runnable example only after the agent implements and validates the process. Create the actual example YAML after implementation and document every configurable leaf with purpose, type/range, available choices and likely tradeoff where known. Include commented-out optional sections. Generic optimizer/scheduler kwargs remain delegated to their upstream factories; the example need not enumerate every keyword of every third-party optimizer.

Validate all Gen2 keys strictly, including nested keys, and reject unknown/misspelled fields. Validate native settings through native schemas plus the v1 compatibility rules. A flag that changes the mathematical contract must not be silently ignored. Produce resolved defaults and an explanation for each rejected incompatibility. Changing logging frequency or memory budgets does not change the mathematical specification; changing targets, masks, time conventions, loss reductions or branch roles does.

`train.validation_config`, when supplied, uses native `ValidationConfig`/item parsing and native deterministic image preprocessing/latent encoding. Gen2 must recompute learned features and use its own branch-aware evaluation, rather than reuse a stale native cached styled embedding or stock loss. Validation image hashes must be disjoint from training hashes when duplicate rejection is enabled. V1 does not automatically split or modify the user's dataset; the example's validation list is optional and user-provided. Fixed training probes are labeled training probes even when no held-out images are available.

```yaml
---
job: extension
config:
  name: "gen2_style_v1"  # Native output/run name; use a new name for a new experiment.
  process:
    - type: "gen2_trainer"  # Exact new extension uid.
      training_folder: "output"  # Native output root; Gen2 records live under its run folder.
      device: "cuda:0"  # V1: one process, one GPU; no automatic multi-GPU fallback.
      trigger_word: "<gen2style>"  # Arbitrary reserved literal; no named style is required.

      network:
        type: "lora"  # V1 fixed family; do not select LoKr/DoRA or merge training weights.
        linear: 32  # Integer >=1, <= smallest target dimension. More rank costs memory/capacity.
        linear_alpha: 32  # Positive float; alpha/rank scales each residual exactly once.
        dropout: 0.0  # Locked to zero in v1; rank/module dropout must also remain zero.

      datasets:
        - folder_path: "/ABSOLUTE/PATH/TO/STYLE_IMAGES"  # Replace; all directories share one style.
          caption_ext: "txt"  # Native matching caption extension; captions describe image content.
          caption_dropout_rate: 0.0  # Locked to zero; do not erase positive content conditioning.
          token_dropout_rate: 0.0  # Locked to zero for paired prompt identity.
          shuffle_tokens: false  # Locked false; do not shuffle JSON or change paired semantics.
          cache_latents_to_disk: true  # Native cache. false trades disk use for repeated VAE work.
          cache_text_embeddings: false  # Locked false; learned features change over the run.
          resolution: [1024]  # Native bucket target(s); multiples supported by the loaded backend.
          num_repeats: 1  # Native integer >=1; changes exposure, not optimizer-step definitions.
          num_workers: 0  # Reference resume mode. >0 needs documented loader replay limitations.
          # Other native data options may be used only when they preserve the paired-state contract.

      train:
        batch_size: 1  # Native microbatch size, integer >=1; increase if memory permits.
        steps: 3000  # Committed Gen2 logical updates; MUST equal the three stage lengths below.
        gradient_accumulation_steps: 1  # Integer >=1; effective batch is the sum of microbatch sizes.
        train_unet: true  # Trains only the constructed LoRA; original DiT weights stay frozen.
        train_text_encoder: false  # Original encoder stays frozen; Gen2 owns its masked adapters.
        gradient_checkpointing: true  # Reuse native DiT checkpointing; memory/compute tradeoff.
        dtype: "bf16"  # Native bf16 recommended; float32/fp16 require backend and gradient checks.
        noise_scheduler: "flowmatch"  # V1 fixed training family.
        timestep_type: "linear"  # V1 fixed finite table; other native distributions are not v1.
        num_train_timesteps: 1000  # Integer >=2; table spans native 1000 to 1 regardless of N.
        content_or_style: "balanced"  # V1 uniform choice over ALL table indices, not cubic bias.
        loss_type: "mse"  # V1 velocity MSE; no additional native loss may be silently mixed in.
        cfg_scale: 1.0  # Training has no CFG. Image sampling has its separate guidance_scale.
        do_cfg: false  # Locked false for all teacher/student training forwards.
        pred_scaler: 1.0  # Locked identity; no extra velocity scaling.
        noise_offset: 0.0  # Locked off; also disable other native noise corrections/augmentations.
        noise_multiplier: 1.0  # Locked identity, so the target uses standard Gaussian epsilon.
        cache_text_embeddings: false  # Locked false, including native sample-feature caches.
        unload_text_encoder: false  # Locked false; A updates need a differentiable encoder path.
        optimizer: "adamw8bit"  # ANY optimizer accepted by the installed native factory is valid.
        lr: 1.0e-4  # Inherited D LR. Prodigy/DAdaptation may reinterpret rates natively; record it.
        optimizer_params:
          weight_decay: 0.0  # Passed to the native factory; zero is the v1 default for all families.
          # betas: [0.9, 0.999]  # Example for compatible Adam-family optimizers only.
        lr_scheduler: "constant"  # Any compatible native scheduler; clock is optimizer-local.
        lr_scheduler_params: {}  # Pass native kwargs; derive local horizons without resetting phases.
        max_grad_norm: 1.0  # Native clipping policy per active family; positive, or native disable value.
        skip_first_sample: false  # Sample initialization for a useful visual reference.
        disable_sampling: false  # true skips images, but MUST NOT disable numeric recording/probes.
        ema_config:
          use_ema: false  # V1 locked off; partial EMA across only one component is invalid.

        # Optional held-out images. Use native schema and preprocessing; Gen2 handles evaluation.
        # Do not point these at training images and then label them held-out.
        # validation_config:
        #   resolution: 1024
        #   validate_every_n_steps: 250  # Committed Gen2 updates, not microbatches.
        #   validation_sigmas: [0.05, 0.25, 0.5, 0.75, 1.0]  # Native/toolkit noise fractions in [0,1].
        #   validation_items:
        #     - image_path: "/ABSOLUTE/PATH/TO/HELD_OUT_IMAGE.png"
        #       prompt: "Content caption for this held-out target-style image"

      model:
        name_or_path: "ideogram-ai/ideogram-4-fp8"  # Native model ID or local model directory.
        arch: "ideogram4"  # Only backend implemented by v1; reject an unsupported arch.
        dtype: "bf16"  # Native model compute/load precision; log actual component precision too.
        quantize: true  # Native DiT quantization; must preserve A-step gradients to its input features.
        qtype: "qfloat8"  # A native supported quantizer; no Gen2 quantization implementation.
        quantize_te: false  # Conservative encoder default; enabling needs actual A-gradient validation.
        qtype_te: "qfloat8"  # Only used if quantize_te=true; native choices/dependencies apply.
        low_vram: false  # Native placement option, not permission to detach A-step computations.
        layer_offloading: false  # Optional native offload; validate frozen-weight input backpropagation.
        compile: false  # Locked off in v1; branch/context-safe compilation is a later optimization.
        unconditional_lora_path: null  # Optional EXISTING frozen native unconditional adapter only.
        model_kwargs:
          text_encoder_path: "Qwen/Qwen3-VL-8B-Instruct"  # Native encoder loader path; match backend.
          max_text_length: 2048  # Total original + M soft positions; require M < limit <=2048 in v1.
        # Do not set pretrained/assistant/inference style LoRAs; v1 starts a fresh diffusion LoRA.

      save:
        dtype: "float32"  # Native inference-export dtype; resume always preserves float32 masters.
        save_every: 250  # Integer >=1 committed updates; also save zero, boundaries and final.
        max_step_saves_to_keep: 4  # Native rolling retention; protect Gen2 boundary checkpoints.
        push_to_hub: false  # Publishing is not part of implementing or validating this pipeline.

      sample:
        sampler: "flowmatch"  # Reuse native Ideogram Euler/sigma utilities with explicit Gen2 roles.
        sample_every: 250  # Frequent-preview interval in committed updates; integer >=1.
        sample_start_step: 0  # Native preview start; nonnegative committed-update count.
        width: 1024  # Backend-supported dimensions; do not silently change them on OOM.
        height: 1024
        prompts:  # Content-only examples; Gen2 selects styled/neutral modes explicitly.
          - "A person reading beside a window with a cat sleeping on the table"
          - "A transparent mechanical device with many interlocking moving parts"
        neg: ""  # V1 fixed empty; Ideogram CFG unconditional input is image-only.
        seed: 42  # First preview seed; repeated verbatim for every ablation.
        walk_seed: false  # Locked false; comparisons need matched initial noise.
        guidance_scale: 7.0  # Native inference CFG; >=0. Higher values can change style/content balance.
        sample_steps: 30  # Integer >=1; hold fixed across comparisons, more steps cost inference time.

      gen2:
        schema_version: "1.0.0"  # Must equal the implemented v1 schema.
        spec_path: "docs/specs/gen2_trainer_v1.md"  # Put an unchanged copy of this file here.
        expected_spec_sha256: null  # null: hash and pin at creation; set digest to enforce a known copy.

        execution:
          training_seed: 20260911  # Integer >=0; isolate training from diagnostics/preview RNG.
          diagnostic_seed: 314159  # Integer >=0; fixed probe/coordinate sampling RNG.
          deterministic_algorithms: false  # true requests framework determinism; unsupported ops error.
          encoder_gradient_checkpointing: true  # Non-reentrant, preserving masks/context and autograd.
          dit_attention_backend: "native"  # native|flash via original set_attention_backend; validate it.
          error_policy: "abort"  # V1 only option; no silent OOM or nonfinite-update skipping.

        conditioning:
          num_tokens: 4  # Integer 1..32 and less than total token budget; more positions add capacity/cost.
          initializer_text: "style"  # Generic shared initialization only; NEVER a required target name.
          initializer_jitter: 0.01  # Float in [0,1]; 0 starts identical vectors, larger values diversify.
          initializer_seed: 271828  # Same recipe/seed may be reused for every style.
          adapter_rank: 4  # Integer >=1 and <= each targeted projection's smaller dimension.
          adapter_alpha: 4.0  # Positive; applied exactly once as alpha/rank.
          overflow_policy: "error"  # V1 only option; report overlength prompts instead of truncating.
          # Fixed v1: suffix after original chat tokens, all decoder MLP down_proj targets,
          # no vocabulary update, fixed forward norms, no target-style semantic anchor or cosine loss.

        losses:
          neutral_weight: 1.0  # >=0; higher favors base preservation, may constrain achievable styling.
          text_adapter_weight: 1.0e-4  # >=0; higher discourages large suffix-adapter residuals.
          gate_center_weight: 1.0e-3  # >=0; higher keeps g near one during calibration.
          gate_smoothness_weight: 1.0e-4  # >=0; higher penalizes rapid changes with noise time.
          # Setting a weight to zero is an explicitly recorded ablation, not an unreported default.

        phases:
          warmup_updates: 450  # Integer >=0, D only with initialized fixed conditioning.
          refinement_updates: 2100  # Integer >=0, deterministic alternating cycle below.
          calibration_updates: 450  # Integer >=0, gates only; all three must sum to train.steps.
          diffusion_updates_per_cycle: 4  # Integer >=1; these D updates precede A updates each cycle.
          conditioning_updates_per_cycle: 1  # Integer >=1; each A update steps E and T once.

        optimizers:
          # null scalar inherits train.*; maps merge over a COPY of the train.* map.
          # Defaults below change only LR. Every role can use ANY native optimizer and native kwargs.
          diffusion:
            optimizer: null
            lr: null  # Inherit 1e-4; scheduler horizon 2130 in this example.
            optimizer_params: {}
            lr_scheduler: null
            lr_scheduler_params: {}
          embedding:
            optimizer: null
            lr: 1.0e-3  # Default for conventional optimizers; horizon 420 here.
            optimizer_params: {}
            lr_scheduler: null
            lr_scheduler_params: {}
          text_adapter:
            optimizer: null
            lr: 1.0e-5  # Lower default for contextual encoder corrections; horizon 420 here.
            optimizer_params: {}
            lr_scheduler: null
            lr_scheduler_params: {}
          gates:
            optimizer: null
            lr: 1.0e-3  # Default gate LR; horizon 450 here. Larger rates may saturate tanh.
            optimizer_params: {}
            lr_scheduler: null
            lr_scheduler_params: {}
          # Optional example: change ONE role without changing the factory implementation.
          # In gates above, replace optimizer/lr with:
          # optimizer: "prodigy"
          # lr: 1.0
          # optimizer_params: {weight_decay: 0.0}
          # Use valid upstream options; Gen2 must record native automatic LR adaptation.

        gates:
          amplitude: 0.5  # rho, strictly between 0 and 1; g lies in (1-rho, 1+rho).
          regularization_grid_points: 65  # Integer >=5, linspace [0,1], both endpoints included.
          # Fixed v1: four cubic Bernstein coefficients per actual block; zero initialization.
          # To omit calibration as an ablation, set calibration_updates=0 and adjust train.steps.

        inference:
          missing_trigger_policy: "learned_neutral"  # learned_neutral|base_bypass; see Section 8.
          lora_strength: 1.0  # Float >=0; inference only. Training strength is exactly one.
          # strength=0 still leaves learned conditioning present when the trigger is present.
          # It is not identical to the base mode, which also removes learned conditioning.

        data:
          content_groups_file: null  # Optional CSV: relative_path,group. Labels only; no new sampler.
          reject_train_validation_duplicates: true  # bool; compare image hashes when validation exists.
          # Group names are user supplied. Missing labels are "unassigned", not guessed easy/hard.

        diagnostics:
          enabled: true  # V1 fixed true for core recording; never silently turn it off.
          activation_every: 100  # Integer >=1; larger reduces all-module activation-summary overhead.
          gradient_probe_every: 250  # Integer >=1; isolated loss-gradient comparisons, no updates.
          gradient_probe_examples: 1  # Integer >=1, <= numerical probe example count.
          gradient_probe_taus: [0.25, 0.75]  # Nonempty unique floats in (0,1], fixed for the run.
          gradient_probe_max_coordinates: 65536  # Per family; 0=full, >0=uniform sampled estimates.
          gate_log_every: 100  # Integer >=1; all blocks on the fixed grid; also every boundary.
          spectra_every: 500  # Integer >=1; small-matrix QR/SVD, also at zero and phase boundaries.
          full_update_norm_every: 0  # 0=off; positive interval requests exact before/after snapshots.
          parameter_sample_elements_per_family: 65536  # Integer >=1; all entries for smaller families.
          tensor_memory_budget_mb: 512  # Positive additional diagnostic-buffer budget, not total VRAM.
          time_bin_edges: [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]  # Strictly increasing, covers [0,1].
          prefix_atol: 0.001  # Positive absolute tolerance for real-backend prefix comparison.
          prefix_rtol: 0.01  # Positive relative tolerance; do not loosen without explaining evidence.
          probes:
            every: 250  # Integer >=1; full 2x2 velocity interaction probes, also boundaries.
            num_examples: 4  # Integer >=1; select fixed examples once using diagnostic_seed.
            source: "training"  # training|validation; validation requires real held-out items above.
            taus: [0.05, 0.25, 0.5, 0.75, 1.0]  # Unique toolkit noise fractions in (0,1].
            save_fixed_latent_packet: true  # V1 fixed true for reproducible numerical comparison.
          tensor_dumps:
            enabled: false  # Optional raw diagnostics; scalar/provenance recording is always retained.
            names: ["suffix_features", "probe_velocities"]  # Allowed names; no arbitrary module dumps.
            max_packets: 4  # Integer >=0 total retained optional packets per run.
            max_total_mb: 256  # Positive total optional-dump budget; explicit skip reason on limit.

        recording:
          flush_every_updates: 10  # Integer >=1; always flush on save/boundary/error.
          rotate_mb: 64  # Positive size of a JSONL segment before rotation.
          compression: "none"  # none|gzip; gzip closed rotated segments, not the active append file.
          writer_queue_records: 4096  # Integer >=1; backpressure, never silently drop core records.
          max_core_recording_mb: 4096  # Positive hard budget excluding weights/images/optional dumps.
          # Reaching the core limit stops cleanly; it does not disable recording and keep training.

        evaluation:
          milestone_every: 1000  # Integer >=1; expanded images, plus every stage boundary/final.
          additional_seeds: [43]  # Appended to sample.seed, deduplicated in order; nonnegative integers.
          prompt_groups: {}  # Optional map: p000: easy, p001: hard. IDs follow native prompt order.
          # Set groups only from actual dataset/experience; examples here have no asserted difficulty.
          preview_modes: ["full", "neutral_lora_on", "base"]  # Nonempty subset of Section 8.1 modes.
          milestone_modes:
            - "full"
            - "neutral_lora_on"
            - "base"
            - "base_with_conditioning"
            - "conditioning_init"
            - "encoder_adapter_off"
            - "tokens_init"
            - "gates_one"
            - "gates_time_mean"
          make_contact_sheets: true  # bool; reuse existing image/grid utilities, retain individual files.

        checkpoint:
          resume_from: null  # Optional complete Gen2 checkpoint directory; not a bare LoRA file.
          strict_resume: true  # V1 fixed true; incompatible restart is a separate user decision/run ID.
          save_at_stage_boundaries: true  # V1 fixed true; preserve interpretable component checkpoints.
          protect_stage_checkpoints: true  # V1 fixed true despite native rolling checkpoint retention.
          save_initial_state: true  # V1 fixed true; zero, initialized tokens and modules are references.

meta:
  name: "[name]"  # Native substitution.
  version: "gen2_trainer_v1"
```

### 12.2 Configuration semantics that must not be guessed

- Four optimizer role maps merge shallowly over a copied native kwargs map; null scalar means inherit. Empty map means no overrides. If an inherited keyword is invalid for a role's selected different optimizer, reject and ask for a compatible configuration. Do not silently remove that keyword. A user may override the common `train.optimizer_params` map and explicitly supply each role's needed kwargs.
- LR scheduler horizons are derived from successful family-update totals, with constructor-specific mapping through the native factory. A role's explicit scheduler kwargs are preserved; contradictory horizon/step settings produce a validation error requiring clarification.
- `tensor_memory_budget_mb` caps additional retained diagnostic buffers. If optional full gradients/snapshots cannot fit, record the method and reason for a sampled result or skipped optional metric. It never authorizes changing a training batch or silently skipping required core rows. Gradient coordinate samples are uniform within each flattened family; scale sampled squared norms by family_size/sample_size, and label the resulting cosine estimate.
- Content-group CSV paths are relative to their declared training dataset directory; for multiple directories require a dataset index column to disambiguate repeated names. Prompt groups use p000, p001, etc., but every prompt also has a content hash so reordered lists remain traceable across runs.
- `save_fixed_latent_packet` is separate from optional raw tensor dumps. The mandatory fixed packet stores z0, epsilon and the probe metadata needed for repetition. Optional packets add selected current suffix features or the 2x2 probe velocities, with total byte/count limits.
- Numerical prefix tolerances apply to production floating-point comparisons only. The small mathematical causal-mask test remains strict. Report a measured baseline numerical-noise comparison before attributing a mismatch to harmless rounding.
- `train.validation_config`, native data transformations, quantization/placement kwargs and optimizer/scheduler kwargs should remain native mechanisms. Other native options that alter v1's loss, gradient ownership, noise, branches, token caching or phase semantics are explicitly incompatible. Produce a compatibility error listing them; do not silently ignore them.
- The strict default has no warm-start style LoRA, target-style literal anchor, base-preservation replay dataset, automatic caption generator, differentiable image-style critic, explicit content router or learned guidance scale. Those are potential future experiments, not unfinished mandatory v1 features.

After coding, enumerate every implemented Gen2 schema leaf and verify it appears in the actual commented example with a documented default and range/choices. No hidden tunable numerical weights, phase fractions, rank choices, gate bounds, sampling intervals or loss transformations may live only in Python. The fixed epsilons and mathematical basis constants specified above are versioned constants, not undocumented tuning knobs.

## 13. Interpreting the first training run

Use the recorded evidence to separate questions, without promising that one run can uniquely settle every hypothesis:

| Observed result | Supported interpretation | Unresolved possibilities |
|---|---|---|
| Missing A-path gradients, altered frozen prefix, disconnected gates or incorrect branch flags | The implementation does not realize v1. Fix before interpreting style quality. | Scientific usefulness after correction. |
| L_N falls and LoRA-on neutral images approach base images | Neutral conditional preservation works on measured states/prompts. | Robustness elsewhere; helpful style with the suffix. |
| Style improves but conditioning_init is similar to full | The package may mostly use suffix presence/diffusion weights; learned conditioning has limited demonstrated additional value. | Whether more conditioning updates or another objective would help hard prompts. |
| Full outperforms initialization on hard prompts with preserved content | Evidence for useful learned conditioning in this package. | Matched-budget superiority and transfer to other styles. |
| Learned gates improve L_S/L_N but not visual outcomes | Reconstruction allocation did not deliver the desired perceptual improvement. | Loss mismatch, bounds, scalar capacity or coverage. |
| Gates outperform g=1 but not constant per-block means | Block rescaling has evidence; time dependence has little additional demonstrated value. | Better temporal parameterizations or data. |
| Easy scenes improve but unseen scenes remain weak | The controlled style route still has a transfer limitation. | Data coverage, representation capacity, optimization and objective mismatch. |

Do not assert universal style activation, a percentage of stored knowledge accessed, or complete separation of style and content. Do not blame the user's original frozen-LoRA Phase A on simultaneous diffusion updates; that historical condition was explicitly corrected. A later matched-budget training comparison is required for strong performance claims. The purpose of this implementation is a faithful, inspectable candidate whose failures and successes generate usable evidence.

## 14. Research provenance and limits

The exact combined pipeline, objectives, coefficients and defaults in this document are a proposed v1. None of the following sources validates the complete combination on Ideogram:

- [Textual Inversion — Gal et al., 2022](https://arxiv.org/abs/2208.01618): image-supervised learned concept/style embeddings; motivates continuous representations without a precise target-style name.
- [TextBoost v2 — Park et al., 2026 revision](https://arxiv.org/html/2409.08248v2): masked causal text adaptation and extended conditioning; motivates preserving the prefix. Its tested architecture and mask placement are not copied blindly into Qwen.
- [Directional Textual Inversion — Kim et al., 2025](https://arxiv.org/html/2512.13672v1): motivates monitoring and controlling token magnitude. V1 uses explicit normalized reparameterization with native optimizers, not a claim to reproduce that paper's optimizer or all its theory.
- [T-LoRA — Soboleva et al., v2](https://arxiv.org/html/2507.05964v2), [FouRA — Borse et al., 2024](https://arxiv.org/html/2406.08798v1): precedents for time/input-adaptive low-rank behavior. The scalar Bernstein gate here is a separate design choice.
- [Ideogram 4 technical details](https://ideogram.ai/blog/ideogram-4.0/): architecture, intermediate encoder features, flow matching and asymmetric CFG; the actual imported code and loaded checkpoint must be verified as described above.

The native code references in Section 2 are integration evidence. The neutral preservation objective, gradient ownership, loss reductions, gate formula and diagnostic intervention definitions are specified directly here so that a coding agent does not need to infer missing mathematics from those papers.

**End of immutable gen2_trainer v1 specification. Implementation notes and subsequent decisions belong in separate files.**
