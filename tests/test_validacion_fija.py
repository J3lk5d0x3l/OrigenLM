"""La validación no debe variar cuando no se modifican los pesos."""
import random
import tempfile
import unittest
from pathlib import Path

import torch

from origen.corpus import preparar_corpus
from origen.model import ModelConfig, OrigenLM
from train_masivo import evaluate, read_shards


class FixedValidationTests(unittest.TestCase):
    def test_same_windows_give_same_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fuentes"
            source.mkdir()
            # Forzar suficientes fuentes para obtener ambas particiones.
            for index in range(70):
                (source / f"doc{index}.txt").write_text(
                    f"Capítulo {index} sobre Python y Rust. " * 9, encoding="utf-8")
            destination = Path(directory) / "prepared"
            preparar_corpus(source, destination)
            shards, _ = read_shards(destination, 24)
            torch.manual_seed(11)
            model = OrigenLM(ModelConfig(dim=32, n_layers=1, n_heads=4,
                                          hidden_dim=64, max_seq_len=32))
            loss1 = evaluate(model, shards["val"], 24, 4,
                             random.Random(9999), torch.device("cpu"))
            loss2 = evaluate(model, shards["val"], 24, 4,
                             random.Random(9999), torch.device("cpu"))
            self.assertAlmostEqual(loss1, loss2, places=6)
            self.assertTrue(model.training)  # restaurar modo para entrenar después


if __name__ == "__main__":
    unittest.main()
