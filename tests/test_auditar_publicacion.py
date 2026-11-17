"""Regresión del detector de secretos (sin usar claves auténticas)."""
import tempfile
import unittest
from pathlib import Path
import subprocess

from scripts.auditar_publicacion import sensitive_filename, secret_signatures, scan_current, scan_history


class AuditPublicationTests(unittest.TestCase):
    def test_protected_paths_and_template(self):
        for path in (".env", ".env.production", "checkpoints/origen.pt",
                     "dataset/preparado_v2/train.bin", "id_ed25519", "private.pem"):
            self.assertTrue(sensitive_filename(path), path)
        self.assertFalse(sensitive_filename(".env.example"))
        self.assertFalse(sensitive_filename("dataset/chat_v2.jsonl"))

    def test_signatures_do_not_print_secret(self):
        self.assertTrue(secret_signatures("-----BEGIN OPENSSH PRIVATE KEY-----"))
        self.assertTrue(secret_signatures("AKIA" + "A" * 16))
        self.assertFalse(secret_signatures("configurar variable de entorno API_KEY"))

    def test_github_history_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            def git(*args):
                return subprocess.run(["git", "-C", str(folder), *args],
                                      check=True, capture_output=True)
            git("init", "-q")
            git("config", "user.email", "test@example.invalid")
            git("config", "user.name", "Audit Test")
            (folder / ".env").write_text("TOKEN=dummy\n")
            git("add", ".env")
            git("commit", "-qm", "primer commit")
            git("rm", "-q", ".env")
            git("commit", "-qm", "borra archivo")
            self.assertEqual(scan_current(folder), [])
            self.assertTrue(any("ruta privada en historial" in result[2]
                                for result in scan_history(folder)))


if __name__ == "__main__":
    unittest.main()
