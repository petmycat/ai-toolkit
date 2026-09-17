"""V2's sole trainable component: freely optimized input vectors."""
from __future__ import annotations

import torch
from torch import nn


class TokenBank(nn.Module):
    """Numerical vocabulary-statistics initialization, with no semantic seed word.

    A private CPU generator first samples ordinary vocabulary row IDs uniformly
    without replacement, then draws independent per-coordinate Gaussian vectors
    using those rows' population mean/std. Each vector is scaled ONCE to the
    median sampled row norm. Forward returns the unnormalized trainable E.
    """
    def __init__(self, embedding, tokenizer, num_tokens=4, initializer_seed=271828,
                 initialization_sample_size=4096):
        super().__init__()
        if num_tokens < 1 or initialization_sample_size < 2 or initializer_seed < 0:
            raise ValueError("Token initialization requires positive bank size, sample size >=2 and nonnegative seed")
        generator = torch.Generator(device="cpu").manual_seed(initializer_seed)
        size = embedding.weight.shape[0]
        special = set(getattr(tokenizer, "all_special_ids", ()))
        ordinary = sorted({int(value) for value in tokenizer.get_vocab().values()
                           if 0 <= int(value) < size and int(value) not in special})
        if len(ordinary) < 2:
            raise ValueError("At least two ordinary vocabulary rows are required for random initialization")
        selection = torch.randperm(len(ordinary), generator=generator)[:min(initialization_sample_size, len(ordinary))]
        ids = torch.tensor(ordinary, dtype=torch.long)[selection]
        with torch.no_grad():
            rows = embedding(ids.to(embedding.weight.device)).detach().to("cpu", torch.float32)
        if not bool(torch.isfinite(rows).all()):
            raise ValueError("Sampled ordinary vocabulary embeddings contain nonfinite values")
        mean, std = rows.mean(0), rows.std(0, unbiased=False)
        radius = rows.norm(dim=-1).quantile(.5)
        if not bool(torch.isfinite(radius)) or float(radius) <= 0:
            raise ValueError("Sampled ordinary vocabulary embeddings have no positive typical norm")
        initial = mean + std * torch.randn((num_tokens, rows.shape[1]), generator=generator, dtype=torch.float32)
        norms = initial.norm(dim=-1, keepdim=True)
        if bool((norms == 0).any()) or not bool(torch.isfinite(norms).all()):
            raise ValueError("Random token initialization produced an invalid vector")
        initial = initial * (radius / norms)
        self.E = nn.Parameter(initial)
        self.register_buffer("initial", initial.clone())
        self.register_buffer("sampled_vocabulary_ids", ids)
        self.register_buffer("vocabulary_mean", mean)
        self.register_buffer("vocabulary_std", std)
        self.register_buffer("initial_typical_norm", radius)
        self.register_buffer("initializer_seed", torch.tensor(initializer_seed, dtype=torch.long))

    def forward(self, mode="learned"):
        if mode == "learned":
            return self.E
        if mode == "init":
            return self.initial
        raise ValueError(f"Unknown token-bank view: {mode}")

    def provenance(self):
        return {"procedure": "uniform ordinary-vocabulary rows without replacement; per-coordinate population Gaussian; one-time median-row-norm scaling",
                "generator": "torch.Generator(cpu)", "seed": int(self.initializer_seed),
                "sampled_vocabulary_ids": self.sampled_vocabulary_ids.detach().cpu().tolist(),
                "sample_size": int(self.sampled_vocabulary_ids.numel()), "num_tokens": self.E.shape[0],
                "embedding_dimension": self.E.shape[1], "typical_initial_norm": float(self.initial_typical_norm),
                "forward_normalization": False, "semantic_initializer": None}


def replace_input_embeddings(embedding, ids, soft_positions, bank_indices, bank):
    """Differentiable indexed replacement; repeated occurrences share one bank."""
    inputs = embedding(ids)
    if not soft_positions:
        return inputs
    positions = torch.tensor(soft_positions, device=inputs.device, dtype=torch.long)
    indices = torch.tensor(bank_indices, device=bank.device, dtype=torch.long)
    values = bank.index_select(0, indices).to(inputs.device, inputs.dtype).unsqueeze(0)
    return inputs.index_copy(1, positions, values)
