# Gen2 v2 implementation validation

## Status on 2026-09-17

The standalone activator path is implemented. The complete local Gen2 test suite
passes: **399 tests**, covering both the retained v1 path and the new v2 path.
This establishes the tested software behavior; it does not establish style
quality, GPU memory fit, or execution of the full quantized Ideogram models.

The user will run the smoke and pilot manually on the RTX PRO 6000 96 GB VM.
No GPU training, model-weight download, or production environment modification
was performed during this implementation.

## Local checks

The final command was:

```text
python -B -m pytest extensions/gen2_trainer/tests -q -p no:cacheprovider
399 passed in 8.81s
```

The local environment uses Windows, Python 3.10, and PyTorch 2.8.0+cpu.
Transformers 4.56.1 is older than the production dependency profile, and
bitsandbytes is absent. A previously installed temporary `oyaml` package was
placed on this command's `PYTHONPATH` so the native YAML parsing test ran rather
than skipping. Global dependencies were not changed.

Both examples pass the native config-file parser and strict v2 schema resolver.
Actual `toolkit.config_modules` construction is not verified locally because
the installed torchaudio DLL fails to load with this local PyTorch environment.
Native model execution remains a VM acceptance item.

Tests exercise:

- Version dispatch and rejection of unsupported or conflicting configuration.
- Original-position marker expansion, shared vector indices, protected
  truncation, native chat serialization, and unchanged original tokenization.
- Input-gradient propagation, frozen backbone invariants, and explicit routing
  to a separate original unconditional transformer.
- Example-weighted gradient accumulation, optimizer ownership and actual
  epsilon, finite gradients/state/updates, and committed update counters.
- Package checksums and identities, atomic publication, exact CPU continuation
  of token and AdamW state, and restoration of data/RNG/evaluation state.
- Recovery after final inference export fails: a completed rolling checkpoint
  can recreate the final package without another update or sampling event.
- Independent save/sample intervals, fixed reference reuse and identity checks,
  initialization image aliasing, and evaluation RNG isolation.
- Representative memory-probe selection without advancing training data,
  optimizer state, or training RNG.

Tiny tensor/model fixtures are used where large native models or CUDA would
otherwise be required. A passing fixture is not a real-backend acceptance claim.

## Actual tokenizer and recorded-caption check

A separate read-only check used the locally cached tokenizer from
`Qwen/Qwen3-VL-8B-Instruct`, without loading or downloading model weights, and
the 40 original captions in the previous run's recorded dataset manifest.

- All 40 captions preserve their three marker occurrences. Four shared vectors
  per occurrence produce 12 learned positions, not three independent banks.
- Resulting sequence lengths range from 458 to 3072 tokens.
- One caption, `r1X1dOn9mA2(14).png`, expands to 3247 tokens before truncation;
  175 ordinary tail tokens are omitted. All activator positions and the full
  chat wrapper are preserved. Source captions are not rewritten.
- The unchanged explosion prompt fits without truncation: 1269 tokens in the
  marker-removed base mode and 1281 in the named/init/learned comparisons.
- A synthetic oversized prompt verifies common retained ordinary content across
  comparison modes while retaining all markers and the complete chat wrapper.
- The original tokenizer is unchanged after compilation.

The explosion prompt's UTF-8 SHA-256 is
`b57f5ce3d890147803624284a2a3ff6b337dd1107033aedcc132b297eb5de17f`.

The user-owned specification remains read-only and unchanged. Its tracked byte
copy has the same SHA-256:
`ec3e254c6638f1781faa449114acc17c1cc8aa454e6a5e79a4032b04ec4c7ae2`.
The tracked copy is exempt from Git text conversion so the pinned bytes survive
checkout on the VM.

## Remaining VM acceptance

Follow [the v2 run guide](v2.md), starting with the four-update smoke.
It runs caption preflight before loading model weights, checks native no-marker
features and ordinary-token gradient connectivity, and probes selected large
image/text workloads before committing the first update.

The VM must establish:

- Real quantized Qwen and conditional DiT input-gradient execution, finite
  nonzero learned-vector gradients/updates, and frozen-weight integrity.
- Separate original unconditional checkpoint loading and image sampling.
- Memory use for representative full-resolution, long-caption batches and
  sampling, with no silent skipping or changes to the configured experiment.
- Actual package reload and continuous-versus-resumed state/image parity using
  the opt-in `acceptance` command in the run guide.
- Actual AdamW8bit behavior if that optimizer is selected. The default examples
  use AdamW because the learned bank is small; CPU tests cannot validate the
  bitsandbytes CUDA path.

After mechanical acceptance, the 500-update pilot is a bounded visual test.
The learned activator must be judged against the base, random initialization,
and named-phrase references at matched prompts and seeds. Lower loss or changed
pixels alone do not establish useful style activation or generalization.
