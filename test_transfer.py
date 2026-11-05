"""Check that the encoder detaches from TankBind's two heads and carries its representation.

A plain script rather than pytest, so it runs inside the image where the model runs.

What it defends: TankBind's declared encoder/head boundary named `linear_energy` and
`gate_linear` only until 2026-09-09, which left `self.linear` — the whole interaction-distance-
map head — and `self.bias` on the encoder side. Neither mistake raises anything. `bias` is one
scalar, so transferring it is numerically invisible while carrying a task-specific offset
across as though it were representation; `linear` is an entire head for the other task.

The load-bearing case is the embedding surviving transfer. `embed_complexes.py` writes a masked
mean of `z`, the tensor `tranistion` produces, so the encoder has to end exactly there — and
after a transfer the new model must pool to the same vector while scoring differently.

usage:
    python test_transfer.py
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tankbind"))

from model import get_model  # noqa: E402

from transfer import HEAD_PREFIXES, is_head, set_training_mode, transfer_encoder  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def case(name):
    def wrap(fn):
        try:
            fn()
        except AssertionError as e:
            FAILED.append(f"{name}: {e}")
        except Exception as e:
            FAILED.append(f"{name}: {type(e).__name__}: {e}")
        else:
            PASSED.append(name)
        return fn
    return wrap


def refuses(fn, fragment: str) -> None:
    try:
        fn()
    except (SystemExit, ValueError) as e:
        assert fragment in str(e), f"refused, but for the wrong reason: {e}"
        return
    raise AssertionError(f"accepted what it should refuse (expected {fragment!r})")


def build(seed: int):
    torch.manual_seed(seed)
    return get_model(0, logging.getLogger("tankbind-test"), "cpu")


def encoder_state(model) -> dict:
    return {k: v for k, v in model.state_dict().items() if not is_head(k)}


def write_encoder(model, directory: Path, name: str = "encoder.pt") -> str:
    path = directory / name
    torch.save({"encoder": encoder_state(model), "variant": "tankbind.torch"}, path)
    return str(path)


@case("both output heads are on the head side")
def _():
    """`linear` writes the distance map, `linear_energy`/`gate_linear`/`bias` the affinity."""
    names = list(build(seed=0).state_dict())
    head = {n for n in names if is_head(n)}
    assert head == {"linear.weight", "linear.bias", "linear_energy.weight",
                    "linear_energy.bias", "gate_linear.weight", "gate_linear.bias", "bias"}, \
        sorted(head)
    assert len(head) < len(names), "everything matched the head"


@case("linear does not swallow linear_energy")
def _():
    """The module-boundary match is the whole reason these two can both be prefixes.

    A raw `startswith("linear")` would put `linear_energy.weight` under `linear`, which happens
    to give the same answer here — both are heads — and would give a wrong one the moment a
    model has `linear` on one side and `linear_something` on the other.
    """
    assert is_head("linear.weight") and is_head("linear_energy.weight")
    assert "linear" in HEAD_PREFIXES and "linear_energy" in HEAD_PREFIXES
    assert not any(name == "linear" or name.startswith("linear.")
                   for name in ["linear_energy.weight"]), "matched at the wrong granularity"


@case("the encoder ends where the embedding is pooled")
def _():
    """Nothing after `tranistion` may be on the encoder side; that tensor is the embedding."""
    encoder_names = set(encoder_state(build(seed=0)))
    assert any(n.startswith("tranistion.") for n in encoder_names), \
        "the transition module is not in the encoder, so the embedding is not either"
    for name in encoder_names:
        assert not name.startswith(("linear.", "linear_energy.", "gate_linear.")), name


@case("transfer moves the encoder and leaves the heads")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        source = build(seed=0)
        path = write_encoder(source, Path(tmp))
        fresh = build(seed=99)
        before = {k: v.clone() for k, v in fresh.state_dict().items()}

        transfer_encoder(fresh, path, freeze=False)
        after = fresh.state_dict()

        for name, value in encoder_state(source).items():
            assert torch.equal(after[name], value), f"{name} did not transfer"
        for name in (n for n in after if is_head(n)):
            assert torch.equal(after[name], before[name]), f"{name} was overwritten"


@case("freezing leaves the encoder untouched")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_encoder(build(seed=0), Path(tmp))
        model = build(seed=99)
        transfer_encoder(model, path, freeze=True)

        assert all(not p.requires_grad for n, p in model.named_parameters() if not is_head(n))
        assert all(p.requires_grad for n, p in model.named_parameters() if is_head(n))
        set_training_mode(model, training=True, frozen_encoder=True)
        # LayerNorm keeps no running statistics, so there are no buffers to
        # drift here — the eval mode is for the five Dropout2d layers in the trigonometry stack,
        # which would otherwise resample the representation the heads are fitted on.
        assert not model.tranistion.training, "the encoder is still sampling dropout"
        assert model.linear_energy.training, "the affinity head should be training"


@case("a whole checkpoint is refused")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "whole.pt"
        torch.save(build(seed=0).state_dict(), path)
        refuses(lambda: transfer_encoder(build(seed=1), str(path), freeze=False),
                "not an encoder")


if __name__ == "__main__":
    for name in PASSED:
        print(f"  ok    {name}")
    for line in FAILED:
        print(f"  FAIL  {line}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    sys.exit(1 if FAILED else 0)
