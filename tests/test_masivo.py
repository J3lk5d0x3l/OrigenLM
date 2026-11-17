"""Pruebas de sampling por documento y evaluación antes de entrenar."""
import tempfile
import unittest
from pathlib import Path
import random
import torch

from origen.corpus import preparar_corpus
from origen.model import ModelConfig, OrigenLM
from train_masivo import read_shards, draw_batch, evaluate


class TrainingCorpusTests(unittest.TestCase):
    def test_eval_then_train_same_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "fuentes"
            folder.mkdir()
            for index in range(50):
                (folder / f"d{index}.txt").write_text(
                    (f"Documento técnico {index}: aprender Python y Rust. " * 8), encoding="utf-8")
            out = Path(tmp) / "preparado"
            preparar_corpus(folder, out)
            shards, _ = read_shards(out, 24)
            cfg = ModelConfig(dim=32, n_layers=1, n_heads=4, hidden_dim=64, max_seq_len=32)
            model = OrigenLM(cfg)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            before = {name: weight.clone() for name, weight in model.state_dict().items()}
            loss = evaluate(model, shards["val"], 24, 2, random.Random(6), torch.device("cpu"))
            self.assertTrue(loss > 0)
            for name, value in before.items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]))
            source, target = draw_batch(shards["train"], 24, 2, random.Random(3), torch.device("cpu"))
            self.assertEqual(source.shape, (2, 24))
            self.assertEqual(target.shape, (2, 24))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            _, train_loss = model(source, target)
            train_loss.backward()
            optimizer.step()
            self.assertFalse(torch.equal(before["tokens.weight"], model.state_dict()["tokens.weight"]))


if __name__ == "__main__":
    unittest.main()
