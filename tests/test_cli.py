"""Pruebas de CLI OrigenLM, sin entrenamiento."""
import tempfile
from pathlib import Path
import unittest

from origen import cli
from origen.resources import effective_cpu_count


class OrigenCliTests(unittest.TestCase):
    def test_v2_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            v1 = Path(directory) / "v1.pt"
            v2 = Path(directory) / "v2.pt"
            self.assertIsNone(cli.chosen_chat(v1, v2))
            v1.touch()
            self.assertEqual(cli.chosen_chat(v1, v2), "chat.py")
            v2.touch()
            self.assertEqual(cli.chosen_chat(v1, v2), "chat_origen_v2.py")

    def test_help_and_unknown_command(self):
        self.assertEqual(cli.main(["--help"]), 0)
        self.assertEqual(cli.main(["not-a-command"]), 2)

    def test_cpu_count_positive(self):
        self.assertGreaterEqual(effective_cpu_count(), 1)

    def test_status_does_not_train(self):
        self.assertEqual(cli.main(["estado"]), 0)


if __name__ == "__main__":
    unittest.main()
