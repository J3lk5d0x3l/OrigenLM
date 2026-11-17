"""Pruebas del KV-cache y de RoPE precomputado en OrigenLM v2."""
import unittest

import torch

from origen.tokenizer import ASSISTANT, BOS, SYSTEM, END, USER, encode
from origen.v2 import KVCache, OrigenV2, OrigenV2Config, rope, rope_tables


def tiny_config(**overrides):
    base = dict(dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
                hidden_dim=128, max_seq_len=64)
    base.update(overrides)
    return OrigenV2Config(**base)


def prompt_ids():
    return [BOS, SYSTEM, *encode("Eres Origen."), END,
            USER, *encode("Hola"), END, ASSISTANT]


class TestRopeTables(unittest.TestCase):
    def test_tables_are_reused_and_grow(self):
        first = rope_tables(16, 8, torch.device("cpu"), torch.float32)
        second = rope_tables(16, 8, torch.device("cpu"), torch.float32)
        self.assertTrue(torch.equal(first[0], second[0]))
        longer = rope_tables(16, 32, torch.device("cpu"), torch.float32)
        self.assertEqual(longer[0].size(-2), 32)
        self.assertTrue(torch.allclose(longer[0][..., :8, :], first[0]))

    def test_offset_matches_full_sequence(self):
        torch.manual_seed(0)
        q = torch.randn(1, 2, 6, 8)
        k = torch.randn(1, 2, 6, 8)
        full_q, full_k = rope(q, k)
        tail_q, tail_k = rope(q[..., 4:, :], k[..., 4:, :], offset=4)
        self.assertTrue(torch.allclose(full_q[..., 4:, :], tail_q, atol=1e-6))
        self.assertTrue(torch.allclose(full_k[..., 4:, :], tail_k, atol=1e-6))

    def test_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            rope_tables(15, 4, torch.device("cpu"), torch.float32)
        with self.assertRaises(ValueError):
            rope_tables(16, 0, torch.device("cpu"), torch.float32)
        with self.assertRaises(ValueError):
            rope(torch.randn(1, 1, 2, 8), torch.randn(1, 1, 2, 8), offset=-1)


class TestKVCache(unittest.TestCase):
    def test_append_and_trim(self):
        cache = KVCache()
        self.assertEqual(cache.length, 0)
        keys = torch.randn(1, 2, 3, 8)
        cache.append(keys, keys.clone())
        cache.append(keys, keys.clone())
        self.assertEqual(cache.length, 6)
        cache.trim(4)
        self.assertEqual(cache.length, 4)
        self.assertTrue(torch.equal(cache.keys, torch.cat((keys, keys), dim=-2)[..., -4:, :]))

    def test_incremental_logits_match_full_forward(self):
        torch.manual_seed(0)
        model = OrigenV2(tiny_config()).eval()
        tokens = torch.tensor([prompt_ids()], dtype=torch.long)
        with torch.inference_mode():
            reference, _ = model(tokens)
            caches = model.new_cache()
            pieces = [model(tokens[:, index:index + 1], caches=caches)[0]
                      for index in range(tokens.size(1))]
        incremental = torch.cat(pieces, dim=1)
        self.assertTrue(torch.allclose(reference, incremental, atol=1e-4))

    def test_chunked_prefill_matches_full_forward(self):
        torch.manual_seed(0)
        model = OrigenV2(tiny_config()).eval()
        tokens = torch.tensor([prompt_ids()], dtype=torch.long)
        with torch.inference_mode():
            reference, _ = model(tokens)
            caches = model.new_cache()
            head, _ = model(tokens[:, :5], caches=caches)
            tail, _ = model(tokens[:, 5:], caches=caches)
        self.assertTrue(torch.allclose(reference, torch.cat((head, tail), dim=1), atol=1e-4))

    def test_greedy_generation_identical_with_and_without_cache(self):
        torch.manual_seed(0)
        model = OrigenV2(tiny_config()).eval()
        prefix = prompt_ids()
        cached = model.generate(prefix, max_new_tokens=24, temperature=0, use_cache=True)
        plain = model.generate(prefix, max_new_tokens=24, temperature=0, use_cache=False)
        self.assertEqual(cached, plain)

    def test_generation_beyond_window_slides(self):
        torch.manual_seed(0)
        config = tiny_config(max_seq_len=32)
        model = OrigenV2(config).eval()
        prefix = [BOS, ASSISTANT, *encode("a" * 28)]
        self.assertEqual(len(prefix), config.max_seq_len - 2)
        produced = model.generate(prefix, max_new_tokens=12, temperature=0)
        self.assertGreater(len(produced), 0)

    def test_cache_rejects_training_and_wrong_length(self):
        model = OrigenV2(tiny_config())
        tokens = torch.tensor([prompt_ids()], dtype=torch.long)
        with self.assertRaises(ValueError):
            model(tokens, targets=tokens, caches=model.new_cache())
        with self.assertRaises(ValueError):
            model(tokens, caches=[KVCache()])

    def test_forward_rejects_overflowing_sequence(self):
        model = OrigenV2(tiny_config(max_seq_len=16))
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 17), dtype=torch.long))
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 0), dtype=torch.long))


class TestResidualInit(unittest.TestCase):
    def test_residual_projections_are_scaled_by_depth(self):
        torch.manual_seed(0)
        model = OrigenV2(tiny_config(n_layers=8))
        expected = 0.02 / (2 * 8) ** 0.5
        for block in model.layers:
            self.assertLess(block.attn.out_proj.weight.std().item(), 0.02)
            self.assertAlmostEqual(block.attn.out_proj.weight.std().item(), expected, delta=expected * 0.4)
            self.assertAlmostEqual(block.ffn.down.weight.std().item(), expected, delta=expected * 0.4)

    def test_lm_head_stays_tied_to_embeddings(self):
        model = OrigenV2(tiny_config())
        self.assertIs(model.lm_head.weight, model.tokens.weight)


if __name__ == "__main__":
    unittest.main()
