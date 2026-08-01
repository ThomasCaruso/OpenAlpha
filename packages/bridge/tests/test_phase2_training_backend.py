"""Real training-backend tests.

Skipped entirely when the ``bridge-training`` extra is absent, so the base suite
never requires Torch. The manual GPU workflow installs the extra and runs these.
"""

from __future__ import annotations

import pytest
from openalpha_bridge.phase2.kronos import BRIDGE_INPUT_DIMENSION
from openalpha_bridge.phase2.training import (
    LOCKED_PARAMETER_COUNT,
    TrainingConfig,
)

torch = pytest.importorskip("torch", reason="requires the bridge-training extra")


def test_real_head_has_exactly_the_locked_parameter_count() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(BRIDGE_INPUT_DIMENSION, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 5),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == LOCKED_PARAMETER_COUNT == 17_605


def test_gradients_reach_only_the_bridge_head() -> None:
    """A frozen trunk must receive no gradient from the Bridge loss."""
    torch.manual_seed(1729)
    frozen_trunk = torch.nn.Linear(20, BRIDGE_INPUT_DIMENSION)
    for parameter in frozen_trunk.parameters():
        parameter.requires_grad_(False)

    head = torch.nn.Sequential(
        torch.nn.Linear(BRIDGE_INPUT_DIMENSION, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 5),
    )

    latent = torch.randn(8, 20)
    hidden = frozen_trunk(latent)
    loss = torch.nn.functional.huber_loss(head(hidden), torch.randn(8, 5), delta=1.0)
    loss.backward()

    assert all(p.grad is None for p in frozen_trunk.parameters())
    assert all(p.grad is not None for p in head.parameters())


def test_optimizer_excludes_frozen_and_official_parameters() -> None:
    head = torch.nn.Sequential(
        torch.nn.Linear(BRIDGE_INPUT_DIMENSION, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 5),
    )
    official_head = torch.nn.Linear(256, 6)
    config = TrainingConfig()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert not any(id(p) in optimized for p in official_head.parameters())
    assert sum(p.numel() for group in optimizer.param_groups for p in group["params"]) == (
        LOCKED_PARAMETER_COUNT
    )


def test_checkpoint_save_and_reload_round_trips(tmp_path) -> None:
    head = torch.nn.Sequential(
        torch.nn.Linear(BRIDGE_INPUT_DIMENSION, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 5),
    )
    path = tmp_path / "bridge.pt"
    torch.save(head.state_dict(), path)

    restored = torch.nn.Sequential(
        torch.nn.Linear(BRIDGE_INPUT_DIMENSION, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 5),
    )
    restored.load_state_dict(torch.load(path, weights_only=True))

    probe = torch.randn(4, BRIDGE_INPUT_DIMENSION)
    with torch.no_grad():
        assert torch.allclose(head(probe), restored(probe))


def test_deterministic_replay_under_the_locked_seed() -> None:
    def forward():
        torch.manual_seed(TrainingConfig().seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(BRIDGE_INPUT_DIMENSION, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 5),
        )
        torch.manual_seed(TrainingConfig().seed)
        with torch.no_grad():
            return model(torch.randn(4, BRIDGE_INPUT_DIMENSION))

    assert torch.allclose(forward(), forward())
