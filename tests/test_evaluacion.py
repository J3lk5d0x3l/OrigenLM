"""Comprobaciones de separación de datos y evaluación sin modificar pesos."""
import json
import math
from pathlib import Path
import unittest

from origen.model import ModelConfig, OrigenLM
from scripts.evaluar_checkpoint import evaluate, read_examples


class DatasetEvaluationTests(unittest.TestCase):
    def test_datasets_disjoint(self):
        root = Path(__file__).resolve().parents[1] / "dataset"
        train = read_examples(root / "chat_v2.jsonl")
        heldout = read_examples(root / "evaluacion_v2.jsonl")
        self.assertGreaterEqual(len(train), 100)
        self.assertGreaterEqual(len(heldout), 10)
        training_questions = {x["user"].strip().casefold() for x in train}
        eval_questions = {x["user"].strip().casefold() for x in heldout}
        self.assertFalse(training_questions & eval_questions)

    def test_eval_does_not_modify_weights(self):
        model = OrigenLM(ModelConfig(dim=32, n_layers=1, n_heads=4, hidden_dim=64, max_seq_len=16))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        loss, count = evaluate(model, [{"user": "Hola", "assistant": "¡Buenas!"}])
        self.assertTrue(math.isfinite(loss))
        self.assertGreater(count, 0)
        for key, old in before.items():
            self.assertTrue(old.equal(model.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
