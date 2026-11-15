from __future__ import annotations

import json
import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("torch is unavailable in this optional-runtime environment") from exc

from actionrankwm.training import (
    IGNORE_INDEX,
    cosine_learning_rate,
    dccp_objective,
    encode_response,
    sequence_log_probability,
)


class _ToyTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = False,
    ) -> list[int]:
        self.assertions = (tokenize, enable_thinking)
        tokens = [1]
        for message in messages:
            role = 10 if message["role"] == "system" else 20 if message["role"] == "user" else 30
            tokens.extend([role, *[100 + ord(char) % 31 for char in message["content"]], 2])
        if add_generation_prompt:
            tokens.append(30)
        return tokens


class _MappingToyTokenizer(_ToyTokenizer):
    def apply_chat_template(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return {"input_ids": super().apply_chat_template(*args, **kwargs)}


class TrainingUtilityTests(unittest.TestCase):
    def test_response_encoding_masks_the_exact_prompt(self) -> None:
        tokenizer = _ToyTokenizer()
        messages = [
            {"role": "system", "content": "simulate"},
            {"role": "user", "content": "{}"},
        ]
        response = json.dumps({"ok": True})
        encoded = encode_response(tokenizer, messages, response, max_length=256)
        self.assertTrue(torch.all(encoded.labels[:, : encoded.prompt_tokens] == IGNORE_INDEX))
        self.assertTrue(torch.all(encoded.labels[:, encoded.prompt_tokens :] >= 0))
        self.assertEqual(encoded.target_tokens, encoded.input_ids.shape[1] - encoded.prompt_tokens)
        with self.assertRaises(ValueError):
            encode_response(tokenizer, messages, response, max_length=2)
        mapped = encode_response(
            _MappingToyTokenizer(), messages, response, max_length=256
        )
        torch.testing.assert_close(mapped.input_ids, encoded.input_ids)

    def test_sequence_log_probability_is_length_normalized(self) -> None:
        logits = torch.zeros(1, 4, 3)
        labels = torch.tensor([[IGNORE_INDEX, 1, 2, 0]])
        value = sequence_log_probability(logits, labels)
        self.assertAlmostEqual(float(value.item()), -torch.log(torch.tensor(3.0)).item())

    def test_dccp_objective_prefers_a_larger_policy_margin(self) -> None:
        ce = torch.tensor(1.0)
        low, _ = dccp_objective(
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            ce,
            beta=1.0,
            alpha=0.1,
        )
        high, diagnostics = dccp_objective(
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            ce,
            beta=1.0,
            alpha=0.1,
        )
        self.assertLess(float(high), float(low))
        self.assertEqual(float(diagnostics["preference_accuracy"]), 1.0)

    def test_cosine_schedule_warms_and_decays(self) -> None:
        values = [
            cosine_learning_rate(
                step,
                10,
                base_learning_rate=1e-3,
                warmup_ratio=0.2,
            )
            for step in range(10)
        ]
        self.assertLess(values[0], values[1])
        self.assertAlmostEqual(values[1], 1e-3)
        self.assertLess(values[-1], values[2])


if __name__ == "__main__":
    unittest.main()
