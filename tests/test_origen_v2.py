"""Pruebas mínimas de OrigenLM-OrigenV2 (pesos generados localmente)."""
import unittest

import torch

from origen.v2 import ORIGEN_V2_PRESETS, OrigenV2Config, OrigenOrigenV2
from origen.chat_training import batchify, tokenize_response
from origen.tokenizer import conversation_prefix_limited, DEFAULT_SYSTEM, VOCAB_SIZE


class OrigenV2Tests(unittest.TestCase):
    def test_config_validates_gqa(self):
        with self.assertRaises(ValueError):
            OrigenV2Config(dim=64, n_heads=4, n_kv_heads=3).validate()
        with self.assertRaises(ValueError):
            OrigenV2Config(dim=60, n_heads=4, n_kv_heads=2).validate()
        for data in ORIGEN_V2_PRESETS.values():
            OrigenV2Config(**data).validate()

    def test_forward_backward_assistant_loss(self):
        torch.manual_seed(8)
        cfg = OrigenV2Config(dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
                           hidden_dim=128, max_seq_len=128)
        model = OrigenOrigenV2(cfg)
        example = tokenize_response({"user": "Hola", "assistant": "¡Buenas!"}, 128)
        x, y = batchify([example], torch.device("cpu"))
        logits, loss = model(x, y)
        self.assertEqual(tuple(logits.shape), (1, x.size(1), VOCAB_SIZE))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(model.tokens.weight.grad)
        self.assertIs(model.lm_head.weight, model.tokens.weight)

    def test_generation_context_sliding_and_greedy(self):
        torch.manual_seed(3)
        cfg = OrigenV2Config(dim=64, n_layers=1, n_heads=4, n_kv_heads=2,
                           hidden_dim=128, max_seq_len=64)
        model = OrigenOrigenV2(cfg).eval()
        prefix = conversation_prefix_limited(DEFAULT_SYSTEM, [], "Hola", 64)
        a = model.generate(prefix, max_new_tokens=8, temperature=0)
        b = model.generate(prefix, max_new_tokens=8, temperature=0)
        self.assertEqual(a, b)
        self.assertLessEqual(len(a), 8)
        with self.assertRaises(ValueError):
            model.generate(prefix, max_new_tokens=0)

    def test_unseen_big_context_and_capacity(self):
        cfg = OrigenV2Config(dim=64, n_layers=1, n_heads=4, n_kv_heads=2,
                           hidden_dim=128, max_seq_len=256)
        model = OrigenOrigenV2(cfg)
        tokens = torch.randint(0, VOCAB_SIZE, (1, 220))
        logits, _ = model(tokens)
        self.assertEqual(logits.size(1), 220)


if __name__ == "__main__":
    unittest.main()
