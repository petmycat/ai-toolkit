# Validation record

## Status on 2026-09-15

**Local verification passed: 106 tests. Real VM acceptance is pending.**

The user will manually run the smoke and actual configurations on an RTX PRO
6000 with 96 GB VRAM, using the VM's existing ai-toolkit dependencies, weights
and images. No real Ideogram training, model download, quantized-backbone
gradient acceptance or `adamw8bit` execution was performed on the local machine.
The scientific hypothesis remains untested until those runs and image reviews.

## Local environment and command

Windows, Python 3.10.10, PyTorch 2.8.0, pytest 7.4.4, NumPy 1.26.2,
safetensors 0.4.5. The local Transformers package is 4.56.1 and does not match
this checkout's required production dependency profile. Optional native model
packages are absent, including bitsandbytes. The tests use CPU tensors.

```text
python -B -m pytest extensions/gen2_trainer/tests -q -p no:cacheprovider
106 passed
```

For the native config-file parsing test, `oyaml==1.0` was installed with
`--no-deps --target` into the temporary directory `gen2-cli-test-deps`, and that
directory was added to `PYTHONPATH` for the test commands only. Global Python
packages and the production environment were not upgraded. Without oyaml,
that single parsing test explicitly skips; ordinary VM dependencies provide it.

## What these tests establish

- Native extension discovery finds the new lazy `gen2_trainer` registration.
  Unrelated optional extensions are filtered in that test; it does not load the
  complete production launcher dependency tree locally.
- All 90 Gen2 schema leaves enforce their types and constraints. Both actual and
  smoke YAMLs contain every leaf. Native config parsing resolves environment/name
  substitutions and scientific notation without importing Torch or model code.
- Float64 normalized-token Jacobian/VJP and finite differences agree. Vocabulary
  rows stay unchanged; effective norms and saved initialization survive reload.
- Causal fixtures with nonzero masked adapters preserve original positions;
  deliberately incorrect masking is detected. Packing tests distinguish channels
  and taps, padding, suffix placement and overflow.
- Native LoRA/helper definitions executed directly from checked-out source verify
  alpha/rank scaling, target-map construction, flow time/sign/padding conventions,
  scoped gates, checkpoint recomputation and bf16 residual compute parity with
  fp32 parameter masters. This source isolation avoids importing unrelated
  unavailable packages; it is not real model execution.
- Controlled D/A/G fixtures verify ownership, both A-family finite checks before
  either step, partial-commit failure handling, unequal accumulation and clipping.
  Actual native-factory Adam, AdamW and Adagrad pass bitwise save/reload
  continuation across D/A/G stages. Actual CPU GradScaler unscales both A
  optimizers and updates once per logical update.
- Zero-horizon families never construct schedulers; inactive state survives
  reload. Scheduler horizons and effective native factory kwargs are checked.
- Native-loader seams preserve default stock construction. Gen2 data checks
  preserve exact captions, reject ambiguous overlap/validation leakage, include
  the native time-table endpoints, and replay shuffle/augmentation across an
  epoch boundary with zero loader workers.
- Recording tests cover finite JSON, bounded queues/backpressure, rotation,
  writer/budget failures, resume identity, joined metadata, blank ratings,
  diagnostic RNG isolation, bounded gradient decomposition, actual token regions,
  masked-write leakage, thin-QR spectra and diagnostic exports.
- Complete-package tests cover all four masters, native export versus resume
  precision, optimizer/RNG restoration, checksums, missing/corrupt components,
  immutable-spec mismatch, atomic publication and protected retention. Read-only
  diagnostic export does not modify its source folder.
- CLI comparison utilities compare every tensor and nested state, keep RNG and
  counters exact, and reject incomplete packages before importing models.

## Additional checks

The user-requested visual diagnostics add three explicit routes: tokens only,
full conditioning with unconditional diffusion LoRA strength 0.5, and full
conditioning with unconditional diffusion LoRA strength 1.0. Existing modes
supply the other three requested comparisons. Tests execute the actual Gen2
sampling loop with a tiny recording backend and source-isolated native sigma
schedule; they verify matched initial noise, image-only unconditional features,
independent absolute strengths, shared gates, unchanged components/gradients/RNG,
and restoration after prediction exceptions. A separate source-isolated native
LoRA test checks composition with the frozen native unconditional adapter using
bf16 projection inputs and fp32 Gen2 masters. Config/CLI checks enforce CFG > 1
for the new unconditional diagnostics while preserving legacy mode defaults.
Contact-sheet tests verify prompt/seed rows, stable mode columns, actual-control
labels, and gaps for deduplicated requests. These remain CPU fixture checks,
not actual Ideogram/CUDA image or training acceptance.

The reviewed local 12-update user smoke passes native YAML parsing and strict
Gen2 validation after repairing the JSON prompt's YAML quoting. It retains one
prompt, arbitrary-text initialization and deliberate weight decay 0.999. Its
family horizons are D9/E1/T1/G2. Seven explicit modes (the requested six plus a
base reference), seeds 42/43, and events 0/2/6/10/12 schedule 70 images. The exact
native caption/chat path was checked with the locally cached Qwen tokenizer at
revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`: 1,269 original positions plus
four learned positions fit the 2,048-position budget. No weights were loaded
or downloaded for that check. The private user YAML remains in ignored `config/`.

Both examples passed the **native** config parser and Gen2 resolver:

- Actual configuration: 3,000 logical updates; D=2,130, E=420, T=420, G=450.
- Smoke configuration: six logical updates; D=3, E=2, T=2, G=1.
- Every Gen2 leaf is present in both examples; no missing schema fields.
- Extension and test Python sources parse successfully without creating bytecode.
- The original local specification and committed reference copy have identical
  SHA-256:
  `6880f48649280c02fe7e73bf9b3b244eef4da6ca63da02226bc876cae5df03b8`.

The actual YAML was generated after implementing the modules, from the unchanged
reference contract, then updated to describe the user-approved optimizer scope
and this checkout's positive-only gradient-clipping limit. The reference itself
was not edited. No code, test logs, caches or generated files were placed under
the protected repository `gen2/` directory.

## Required VM acceptance

Follow [README.md](README.md) to edit paths, validate and run the smoke example,
then run the independent acceptance harness. The harness uses actual native
weights and native optimizers. It checks D/A/G changes, real A-feature/output
changes, all 13 real prefix taps, continuous/resumed full state, frozen weight
identities, and live versus reloaded package inference. It writes measured
results to `acceptance.json` and keeps failure evidence.

Do not report this implementation as having passed production acceptance before
that evidence exists. Optional optimizers and quantization/offload/precision
profiles require their own matching VM runs. A failure must remain visible;
there is no automatic dependency downgrade, precision change, missing-gradient
bypass, tolerance relaxation or mocked replacement.
