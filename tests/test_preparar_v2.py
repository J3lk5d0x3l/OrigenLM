"""Tokenizer propio y split de validación del corpus V2."""
import json
import tempfile
import unittest
from pathlib import Path
from origen.tokenizer_v2 import train_tokenizer, load_tokenizer, encode_document
from scripts.preparar_v2 import prepare


class PreparaV2Tests(unittest.TestCase):
    def test_roundtrip(self):
        tk = train_tokenizer(["¡Hola Colombia! Python y Rust. " * 12,
                              "def sumar(a,b):return a+b\n" * 12], vocab_size=512)
        phrase = "Bogotá, ñ, 😀"
        self.assertEqual(tk.decode(encode_document(tk, phrase)[1:-1]), phrase)

    def test_split_and_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, original, new = root / "fuentes", root / "prev", root / "v2"
            source.mkdir()
            original.mkdir()
            (source / "a.txt").write_text("Programación, Colombia y ciencia. " * 20)
            (source / "b.txt").write_text("Este fragmento es reservado para evaluar. " * 20)
            docs = [{"path":"a.txt", "split":"train", "sha256":"a"*64},
                    {"path":"b.txt", "split":"val", "sha256":"b"*64}]
            (original / "meta.json").write_text(json.dumps({"format":"origen-byte-v1", "docs":docs}))
            data = prepare(original / "meta.json", source, new, vocab=512)
            self.assertEqual(data["total_documents"], 2)
            self.assertEqual(data["format"], "origen-bpe-v2")
            self.assertEqual(data["docs"][0]["split"], "train")
            self.assertEqual(data["docs"][1]["split"], "val")
            self.assertEqual((new / "train.bin").stat().st_size, data["splits"]["train"]["bytes"])
            self.assertEqual((new / "val.bin").stat().st_size, data["splits"]["val"]["bytes"])
            self.assertIsNotNone(load_tokenizer(new / "tokenizer.json"))
            self.assertFalse((original / "train.bin").exists())

    def test_never_overwrite_v1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, original = root / "fuentes", root / "preparado"
            source.mkdir()
            original.mkdir()
            (original / "meta.json").write_text(json.dumps({"format":"origen-byte-v1", "docs":[
                {"path":"a", "split":"train"}, {"path":"b", "split":"val"}]}))
            with self.assertRaisesRegex(ValueError, "no puede sobrescribir"):
                prepare(original / "meta.json", source, original)


if __name__ == "__main__":
    unittest.main()
