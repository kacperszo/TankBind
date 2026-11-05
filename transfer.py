"""Move a trained encoder into a fresh TankBind, and freeze it if asked.

Cutting a checkpoint into encoder and head needs no model code; loading one back does, so it
lives here.

**Where TankBind's boundary is, and why the boundary is where it is.** `forward` returns two
things and there are therefore two heads:

    b = self.linear(z);            y_pred = b[z_mask].sigmoid() * 10        the distance map
    pair = self.gate_linear(z).sigmoid() * self.linear_energy(z) * z_mask
    affinity_pred = self.leaky(self.bias + pair.sum((-1, -2)))              the affinity

Until 2026-09-09 the declared head was `linear_energy` and `gate_linear` only, which left
`self.linear` — the entire interaction-distance-map head — and `self.bias` on the encoder side.
`bias` is one scalar, so transferring it is numerically invisible while carrying a task-specific
offset across as though it were representation; `linear` is a whole head for the other task.

The encoder is everything up to and including `tranistion`, which is exactly what produces `z`
— and `z` masked-meaned is what `embed_complexes.py` writes. So the encoder boundary and the
embedding boundary name the same cut, which is what makes a probe result predict what a
fine-tune starts from.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: The two output heads, by parameter-name prefix. Everything else is the
#: encoder.
#:
#: `linear` and `linear_energy` are two different modules, and only the module-boundary match
#: below tells them apart — a raw string prefix would make `linear` swallow `linear_energy`,
#: which is why `is_head` is written the way it is rather than with `startswith`.
HEAD_PREFIXES = ("linear", "linear_energy", "gate_linear", "bias")


def is_head(name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in HEAD_PREFIXES)


def read_encoder(path: str) -> dict[str, torch.Tensor]:
    """Load an encoder file, accepting either shape it arrives in, always strictly."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get("encoder"), dict):
        payload = payload["encoder"]
    if isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict):
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a state dict, got {type(payload).__name__}")
    return {k: v for k, v in payload.items() if torch.is_tensor(v)}


def transfer_encoder(model: nn.Module, path: str, freeze: bool = False) -> dict[str, object]:
    """Overlay an encoder onto a fresh model, leaving both heads at their initialisation."""
    encoder = read_encoder(path)
    target = model.state_dict()

    head_keys = [k for k in encoder if is_head(k)]
    if head_keys:
        raise SystemExit(
            f"{path} carries {len(head_keys)} head tensors ({head_keys[:3]}). That is a whole "
            f"checkpoint, not an encoder — pass a file holding encoder tensors only."
        )
    unknown = [k for k in encoder if k not in target]
    if unknown:
        raise SystemExit(f"{path}: {len(unknown)} tensors have no slot in this model: {unknown[:5]}")
    mismatched = [k for k, v in encoder.items() if target[k].shape != v.shape]
    if mismatched:
        raise SystemExit(
            f"{path}: shape mismatch on {len(mismatched)} tensors: "
            + ", ".join(f"{k} {tuple(encoder[k].shape)} into {tuple(target[k].shape)}"
                        for k in mismatched[:3])
            + ". The encoder was trained at a different width."
        )
    uncovered = [k for k in target if k not in encoder and not is_head(k)]
    if uncovered:
        raise SystemExit(
            f"{path}: {len(uncovered)} model tensors are neither in the encoder nor in the "
            f"head — the boundary is wrong: {uncovered[:5]}"
        )

    model.load_state_dict({**target, **encoder}, strict=True)
    moved = sum(v.numel() for v in encoder.values())
    print(f"transferred {len(encoder)} tensors / {moved:,} params from {path}")

    frozen = 0
    if freeze:
        for name, param in model.named_parameters():
            if not is_head(name):
                param.requires_grad_(False)
                frozen += param.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"froze {frozen:,} encoder params; {trainable:,} trainable in the two heads")

    return {"init_encoder": path, "transferred_tensors": len(encoder),
            "transferred_params": moved, "frozen_params": frozen}


def set_training_mode(model: nn.Module, training: bool, frozen_encoder: bool) -> None:
    """Keep a frozen encoder deterministic.

    TankBind's trigonometry stack is wrapped in `nn.Dropout2d(p=0.25)` at every one of the five
    rounds, so leaving it in train mode means the heads learn against a representation resampled
    each step — a quarter of the interaction tensor zeroed differently every time. That is a
    different experiment from a linear probe on the frozen embedding.
    """
    if not training or not frozen_encoder:
        model.train(training)
        return
    model.eval()
    for name, module in model.named_modules():
        if name and is_head(name) and any(p.requires_grad for p in module.parameters()):
            module.train()
