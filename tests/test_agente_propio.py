"""Garantiza que el agente usa checkpoints originales, no proveedores externos."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from origen.agente import cli
from origen.agente.backends import checkpoint_predeterminado
from origen.agente.config import cargar_config


class LocalAgentTests(unittest.TestCase):
    def test_checkpoint_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("origen_latest.pt", "origen_v2_latest.pt", "origen_masivo.pt"):
                (root / name).touch()
            with patch("origen.agente.backends.ROOT", root.parent):
                self.assertIsNone(checkpoint_predeterminado())
    def test_config_has_no_external_model(self):
        cfg = cargar_config()
        self.assertFalse(hasattr(cfg, "modelo"))
        self.assertFalse(hasattr(cfg, "backend"))
        self.assertFalse(hasattr(cfg, "api_key"))

    def test_tool_calls_require_explicit_user_command(self):
        self.assertIsNone(cli.herramienta_desde_linea("Hola", None))
        self.assertIn("Uso:", cli.herramienta_desde_linea("/usar ", None))
