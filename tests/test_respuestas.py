"""Pruebas del entrenamiento orientado a respuestas de OrigenLM."""
import math
import unittest

import torch

from origen.chat_training import IGNORE_INDEX, batchify, prepare_records, response_nll, tokenize_response
from origen.model import ModelConfig, OrigenLM
from origen.tokenizer import ASSISTANT, EOS, encode


class ResponseTrainingTests(unittest.TestCase):
    def test_only_assistant_response_is_supervised(self):
        source, targets = tokenize_response({"user": "Hola", "assistant": "¡Buenas!"}, 384)
        index = source.index(ASSISTANT)
        self.assertTrue(all(x == IGNORE_INDEX for x in targets[:index]))
        self.assertEqual(targets[index], encode("¡Buenas!")[0])
        self.assertEqual(targets[-1], EOS)
        self.assertEqual(len(source), len(targets))

    def test_bad_context_is_rejected_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "demasiado largo"):
            tokenize_response({"user": "Hola" * 200, "assistant": "Sí"}, 96)

    def test_batch_padding_is_ignored_and_gradients_flow(self):
        cfg = ModelConfig(dim=32, n_layers=1, n_heads=4, hidden_dim=64, max_seq_len=192)
        model = OrigenLM(cfg)
        examples = prepare_records([
            {"user": "Hola", "assistant": "¡Buenas!"},
            {"user": "Gracias", "assistant": "De nada."},
        ], 192)
        source, target = batchify(examples, torch.device("cpu"))
        self.assertEqual(source.shape, target.shape)
        self.assertEqual(int((target != IGNORE_INDEX).sum()), len(encode("¡Buenas!")) + len(encode("De nada.")) + 2)
        _, loss = model(source, target)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(model.tokens.weight.grad)

    def test_response_eval_does_not_update_weights(self):
        model = OrigenLM(ModelConfig(dim=32, n_layers=1, n_heads=4, hidden_dim=64, max_seq_len=128))
        samples = prepare_records([{"user": "Hola", "assistant": "¡Buenas!"}], 128)
        before = {name: p.detach().clone() for name, p in model.state_dict().items()}
        nll, count = response_nll(model, samples)
        self.assertTrue(math.isfinite(nll))
        self.assertGreater(count, 0)
        for name, old in before.items():
            self.assertTrue(old.equal(model.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
