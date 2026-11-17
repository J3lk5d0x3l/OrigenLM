"""Verificación de la biblioteca binaria usada para preentrenar."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

from origen.corpus import preparar_corpus, split_document


class CorpusTests(unittest.TestCase):
    def test_split_deterministic(self):
        self.assertEqual(split_document("1" * 64), split_document("1" * 64))

    def test_prepare_and_validate_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "fuentes"
            corpus.mkdir()
            # Variar el contenido para forzar train/val; no confiar en azar.
            for i in range(60):
                (corpus / f"archivo{i}.txt").write_text(
                    f"Capitulo numero {i}. Python sirve para escribir programas. " * 6,
                    encoding="utf-8")
            output = root / "tokens"
            metadata = preparar_corpus(corpus, output)
            self.assertGreater(metadata["splits"]["train"]["tokens"], 256)
            self.assertGreater(metadata["splits"]["val"]["tokens"], 128)
            self.assertEqual(metadata["total_documents"], 60)
            for split in ("train", "val"):
                values = (output / f"{split}.bin").read_bytes()
                self.assertEqual(len(values), metadata["splits"][split]["tokens"] * 2)
                self.assertIn(256, struct.unpack("<" + "H" * (len(values)//2), values))
            self.assertEqual(json.loads((output / "meta.json").read_text())["version"], 1)

    def test_rejects_binary_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "data"
            source.mkdir()
            (source / "no.txt").write_bytes(b"\0"*300)
            (source / "a.txt").write_text("Hola mundo "*40)
            (source / "b.txt").write_text("Hola mundo "*40)
            with self.assertRaisesRegex(ValueError, "Corpus insuficiente"):
                preparar_corpus(source, Path(tmp) / "output")
