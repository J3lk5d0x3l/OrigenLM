"""El cierre de V1 debe ser reproducible y nunca sobrescribir originales."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.cerrar_v1 import backup_checkpoint_files, inspect_data


class ClosureV1Tests(unittest.TestCase):
    def test_zip_has_verified_hashes_without_modifying_weights(self):
        with tempfile.TemporaryDirectory() as dirname:
            directory = Path(dirname)
            original = directory / "origen.pt"
            original.write_bytes(b"pesos originales para probar el respaldo")
            target = directory / "export/v1.zip"
            manifest = backup_checkpoint_files({"modelo": original}, target)
            self.assertEqual(original.read_bytes(), b"pesos originales para probar el respaldo")
            self.assertEqual(manifest["files"]["modelo"]["size_bytes"], original.stat().st_size)
            with zipfile.ZipFile(target) as archive:
                self.assertEqual(archive.read("origen.pt"), original.read_bytes())
                self.assertEqual(json.loads(archive.read("manifest.json"))["format"],
                                 "origen-v1-backup-1")

    def test_question_overlap_detection(self):
        with tempfile.TemporaryDirectory() as dirname:
            directory = Path(dirname)
            a = directory / "train.jsonl"
            b = directory / "val.jsonl"
            a.write_text('{"user":"Hola", "assistant":"Hola"}\n', encoding="utf-8")
            b.write_text('{"user":"hola", "assistant":"Buenas"}\n', encoding="utf-8")
            self.assertEqual(inspect_data(a, b)["solapamiento_exacto"], 1)


if __name__ == "__main__":
    unittest.main()
