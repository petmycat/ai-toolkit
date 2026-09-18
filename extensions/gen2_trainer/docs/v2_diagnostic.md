# Post-training v2 diagnostic job

This job investigates a completed standalone activator using the normal
`python run.py config/<file>.yaml` entry point. Its extension type is
`gen2_v2_diagnostic`. Runnable YAML files belong in the repository's `config/`
folder; diagnostic implementation remains in `extensions/gen2_trainer/v2/`.

## Inputs and output

The process takes `source_checkpoint`, `training_folder`, `device`, and a
`diagnostic` mapping. The source is a complete v2 `inference_final` or
`resume_latest` package. Only its learned/initial token tensors and saved
configuration are loaded. Source training optimizer state is never resumed.

Model settings, dataset captions, native resolutions, tokenizer, token placement,
and optimizer hyperparameters come from the source package. Model, tokenizer,
and dataset identities must match the source. Physical diagnostic batch size is
one. `diagnostic.named_phrase` sets an evaluation-only reference; null inherits
the source phrase. A nonempty phrase is required for this four-way comparison.

Use a fresh job name for each execution. Results are written beneath
`<training_folder>/<job_name>/gen2_v2_diagnostic/`. No training checkpoint or
sample image is published. The source package and its parent training run are
protected from output overlap. The repository's `gen2/` folder remains read-only
to the pipeline.

## Measurements

1. **Matched objective measurements:** select distinct existing native image
   batches spread over caption lengths, including the shortest and longest,
   separately for each expanded dataset/resolution. Fix each native crop/latent
   and Gaussian noise draw. Replay the exact same noisy inputs and targets at
   each configured noise fraction with base, initial, learned, and named
   conditioning. The prompt compiler enforces identical retained ordinary
   caption content across these four modes. Loss is FP32 per-example MSE, as in
   v2 training. Reports include paired prediction differences, noise level, and
   resolution breakdowns. This is a training-data subset, not held-out validation.
2. **Controlled descent:** select a small spread of those fixed packets. Run
   fresh temporary AdamW optimization separately from the saved initial bank and
   learned bank, using the source learning rate/betas/epsilon/weight decay. Report
   the identical fixed objective before and after every update. All source token
   values, gradients, flags, and RNG state are restored. This currently requires
   a source trained with `adamw`; another optimizer is explicitly rejected.
3. **Gradient fidelity:** use the shortest selected caption at the smallest
   native resolution, at the configured noise fraction nearest 0.5. Compare a
   repeated configured baseline with encoder checkpointing off, DiT checkpointing
   off, and both off. Replay identical RNG and inputs. Compare gradients, loss,
   predictions, and CUDA memory. Check selected real encoder/DiT projections
   against plain linear operations with the same represented, dequantized frozen
   weights, using captured real input rows. This does not compare the quantized
   model against a different full-precision model.

The supplied diagnostic1 configuration uses four examples per resolution, two
noise seeds, five noise fractions, and two 12-update temporary descents. For the
two-resolution source this means 80 fixed packets and 320 paired diffusion
forwards, plus descent evaluations/backward passes and gradient checks. The
actual counts are printed. Large tensors are kept on CPU between measurements;
the configured storage budget is enforced. No entire model is cloned for
gradient comparisons.

## Files to return for review

Copy the complete diagnostic job output folder. It contains:

- `diagnostic_summary.json`: overall execution and check statuses, paired losses,
  descent results, gradient comparisons, source identity, and explicit limits.
- `paired_losses.jsonl`: every completed matched-input measurement.
- `descent.jsonl`: step-by-step temporary optimization measurements.
- `gradient_fidelity.jsonl`: checkpoint and selected-projection checks.
- `packet_manifest.json`: selected image identities, transforms, fixed noise
  seeds/timesteps, and latent/noise/target hashes.
- `diagnostic_settings.json`, inherited configs, run manifest, caption report,
  dataset manifest, and events: configuration and provenance.

Console progress is printed during packet preparation, objective comparisons,
temporary updates, and gradient checks. Each completed diagnostic scalar record
is flushed to disk. Exceptions write an aborted summary while preserving earlier
evidence. CUDA OOM or unavailable numerical comparisons are inconclusive, never
silently passing. `completed` means the measurements ran, not that the activator
learned a useful style. The summary always leaves `style_acceptance` as
`not_measured`; MSE is not a perceptual style score.

CPU tests validate the orchestration and numerical contracts using small
fixtures. Actual Ideogram/Quanto/CUDA results must come from the VM run.
