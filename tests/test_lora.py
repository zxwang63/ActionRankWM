from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("torch is unavailable in this optional-runtime environment") from exc

from actionrankwm.lora import (
    LoRALinear,
    inject_lora,
    load_adapter,
    lora_parameters,
    save_adapter,
    trainable_parameter_summary,
)


class _ToyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 6, bias=False)
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(6, 4, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.mlp.up_proj(torch.tanh(self.self_attn.q_proj(value)))


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_ToyLayer()])
        self.model.visual = nn.Linear(4, 4, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model.layers[0](value)


class LoRATests(unittest.TestCase):
    def test_injection_is_initially_exact_and_freezes_base(self) -> None:
        torch.manual_seed(4)
        model = _ToyModel()
        value = torch.randn(3, 4)
        expected = model(value).detach().clone()
        specs = inject_lora(model, rank=2, alpha=4.0)
        self.assertEqual(len(specs), 2)
        self.assertTrue(
            isinstance(model.model.layers[0].self_attn.q_proj, LoRALinear)
        )
        torch.testing.assert_close(model(value), expected)
        summary = trainable_parameter_summary(model)
        self.assertGreater(summary["trainable"], 0)
        self.assertFalse(model.model.visual.weight.requires_grad)
        self.assertTrue(all(parameter.requires_grad for parameter in lora_parameters(model)))

    def test_only_adapter_changes_and_roundtrips(self) -> None:
        torch.manual_seed(7)
        model = _ToyModel()
        base_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        inject_lora(model, rank=2, alpha=2.0)
        optimizer = torch.optim.AdamW(lora_parameters(model), lr=0.1)
        value = torch.randn(5, 4)
        loss = model(value).square().mean()
        loss.backward()
        optimizer.step()
        for name, before in base_before.items():
            parameters = dict(model.named_parameters())
            current = parameters.get(name)
            if current is None and name.endswith(".weight"):
                current = parameters.get(name[: -len(".weight")] + ".base.weight")
            self.assertIsNotNone(current, name)
            assert current is not None
            torch.testing.assert_close(current, before)
        expected = model(value).detach().clone()

        with tempfile.TemporaryDirectory() as temporary:
            adapter_dir = Path(temporary) / "adapter"
            save_adapter(model, adapter_dir, metadata={"test": True})
            torch.manual_seed(7)
            restored = _ToyModel()
            load_adapter(restored, adapter_dir)
            torch.testing.assert_close(restored(value), expected)


if __name__ == "__main__":
    unittest.main()
