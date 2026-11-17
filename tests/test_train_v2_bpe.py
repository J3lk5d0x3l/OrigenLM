"""Regresiones del entrenador V2 corregido.

Cubre los cuatro fallos reportados: corte por `--max-minutos` sin guardar,
calendario de LR reiniciado al reanudar, falta de verificación del tokenizador
y ausencia de soporte GPU (resolución de dispositivo y precisión).
"""
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import train_v2_bpe as t

CONFIG_MICRO = {
    "vocab_size": 262, "dim": 16, "n_layers": 1, "n_heads": 4,
    "n_kv_heads": 2, "hidden_dim": 32, "max_seq_len": 32,
}


def _bins(carpeta: Path, tokens: int = 600, vocab: int = 262) -> Path:
    carpeta.mkdir(parents=True, exist_ok=True)
    gen = np.random.default_rng(0)
    for split, n in (("train", tokens), ("val", 200)):
        datos = gen.integers(0, vocab, size=n, dtype=np.int64)
        (carpeta / f"{split}.bin").write_bytes(
            np.asarray(datos, dtype="<u2").tobytes())
    meta = {
        "format": "origen-bpe-v2", "version": 1, "vocab_size": vocab,
        "tokenizer": "tokenizer.json", "tokenizer_sha256": "0" * 64,
        "total_tokens": tokens + 200,
        "splits": {
            "train": {"path": "train.bin", "tokens": tokens, "bytes": tokens * 2},
            "val": {"path": "val.bin", "tokens": 200, "bytes": 400},
        },
    }
    (carpeta / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return carpeta


def _argumentos(root: Path, salida: Path) -> list[str]:
    datos = _bins(root / "datos")
    config = root / "config.json"
    config.write_text(json.dumps(CONFIG_MICRO), encoding="utf-8")
    return ["--datos", str(datos), "--config", str(config), "--salida", str(salida),
            "--ctx", "8", "--batch-size", "2", "--grad-accum", "1",
            "--warmup", "2", "--guardar-cada", "100", "--eval-cada", "100",
            "--device", "cpu", "--precision", "fp32"]


class CalendarioTests(unittest.TestCase):
    def test_warmup_lineal(self):
        self.assertAlmostEqual(t.tasa_aprendizaje(100, 3000, 200, 6e-4), 3e-4)

    def test_coseno_baja_y_respeta_min_ratio(self):
        alta = t.tasa_aprendizaje(300, 3000, 200, 6e-4, 0.1)
        media = t.tasa_aprendizaje(1600, 3000, 200, 6e-4, 0.1)
        final = t.tasa_aprendizaje(3000, 3000, 200, 6e-4, 0.1)
        self.assertGreater(alta, media)
        self.assertGreater(media, final)
        self.assertAlmostEqual(final, 6e-4 * 0.1)

    def test_reanudar_conserva_la_curva(self):
        # Paso 461 de 3000: la tasa debe ser la de la curva original...
        continuacion = t.tasa_aprendizaje(461, 3000, 200, 6e-4, 0.1)
        self.assertGreater(continuacion, 5e-4)
        # ...no la de un warmup reiniciado (paso 1 de un run nuevo = 3e-6).
        reiniciado = t.tasa_aprendizaje(1, 2540, 200, 6e-4, 0.1)
        self.assertGreater(continuacion / reiniciado, 100)

    def test_calendario_invalido(self):
        with self.assertRaises(ValueError):
            t.tasa_aprendizaje(0, 10, 1, 1e-3)
        with self.assertRaises(ValueError):
            t.planificar_pasos(0, 0)


class PresupuestoTests(unittest.TestCase):
    def test_nuevo_empieza_en_cero(self):
        self.assertEqual(t.planificar_pasos(0, 3000), (3000, 3000))

    def test_reanudar_suma_pasos_nuevos(self):
        self.assertEqual(t.planificar_pasos(500, 2500, None, 3000), (3000, 3000))

    def test_reanudar_corto_conserva_el_calendario(self):
        self.assertEqual(t.planificar_pasos(500, 100, None, 3000), (600, 3000))

    def test_amplia_calendario_si_se_pide_mas(self):
        self.assertEqual(t.planificar_pasos(500, 3000, None, 3000), (3500, 3500))

    def test_total_explicito_manda(self):
        self.assertEqual(t.planificar_pasos(500, 10, 2000, 3000), (2000, 2000))
        with self.assertRaises(ValueError):
            t.planificar_pasos(500, 10, 400)


class CompatibilidadTests(unittest.TestCase):
    def setUp(self):
        self.meta = {"format": "origen-bpe-v2", "vocab_size": 8192,
                     "tokenizer_sha256": "a" * 64}
        self.carga = {"config": {"vocab_size": 8192}, "tokenizer_sha256": "a" * 64}

    def test_coincide(self):
        self.assertEqual(t.validar_compatibilidad(self.carga, self.meta, "a" * 64), [])

    def test_tokenizador_distinto(self):
        with self.assertRaises(ValueError):
            t.validar_compatibilidad(self.carga, self.meta, "b" * 64)

    def test_corpus_con_otro_tokenizador(self):
        meta = dict(self.meta, tokenizer_sha256="b" * 64)
        with self.assertRaises(ValueError):
            t.validar_compatibilidad(self.carga, meta, None)

    def test_vocab_distinto(self):
        meta = dict(self.meta, vocab_size=4096)
        with self.assertRaises(ValueError):
            t.validar_compatibilidad(self.carga, meta, None)

    def test_formato_distinto(self):
        with self.assertRaises(ValueError):
            t.validar_compatibilidad(self.carga, {"format": "origen-byte-v1"}, None)

    def test_forzar_degrada_a_aviso(self):
        avisos = t.validar_compatibilidad(self.carga, self.meta, "b" * 64, forzar=True)
        self.assertEqual(len(avisos), 1)
        self.assertIn("forzado", avisos[0])


class DispositivoTests(unittest.TestCase):
    def test_cpu_siempre_disponible(self):
        self.assertEqual(t.resolver_dispositivo("cpu").type, "cpu")
        self.assertEqual(t.resolver_dispositivo("auto").type,
                         "cuda" if torch.cuda.is_available() else "cpu")

    def test_cuda_ausente_falla_claro(self):
        if torch.cuda.is_available():
            self.skipTest("hay GPU disponible")
        with self.assertRaises(SystemExit):
            t.resolver_dispositivo("cuda")

    def test_precision_en_cpu_es_fp32(self):
        self.assertEqual(t.resolver_precision(torch.device("cpu")).nombre, "fp32")

    def test_precision_bf16_y_fp16(self):
        if not torch.cuda.is_available():
            self.skipTest("sin GPU")
        self.assertIn(t.resolver_precision(torch.device("cuda"), "fp32").nombre, {"fp32"})
        self.assertEqual(t.resolver_precision(torch.device("cuda"), "fp16").nombre, "fp16")
        bf16 = t.resolver_precision(torch.device("cuda"), "bf16")
        self.assertEqual(bf16.nombre, "bf16")
        self.assertFalse(bf16.necesita_scaler)


class GuardadoTests(unittest.TestCase):
    def test_escritura_atomica_y_portable(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "sub" / "ck.pt"
            t.guardar_checkpoint(ruta, {"architecture": t.ARQUITECTURA, "step": 7,
                                        "model": {"w": torch.zeros(2, device="cpu")}})
            self.assertTrue(ruta.is_file())
            self.assertFalse(ruta.with_name(ruta.name + ".tmp").exists())
            carga = torch.load(ruta, map_location="cpu", weights_only=True)
            self.assertEqual(carga["step"], 7)
            self.assertEqual(carga["model"]["w"].device.type, "cpu")

    def test_huella_opcional(self):
        self.assertIsNone(t.huella_opcional(Path("no-existe.json")))


class EntrenadorTests(unittest.TestCase):
    def test_limite_de_tiempo_guarda_el_ultimo_paso(self):
        """Regresión: antes se salía sin checkpoint si el paso no era múltiplo."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            salida = root / "ck.pt"
            argumentos = _argumentos(root, salida) + ["--pasos", "5",
                                                      "--max-minutos", "0.0001"]
            with contextlib.redirect_stdout(io.StringIO()) as consola:
                self.assertEqual(t.main(argumentos), 0)
            self.assertIn("se guarda antes de salir", consola.getvalue())
            carga = torch.load(salida, map_location="cpu", weights_only=True)
            self.assertLess(carga["step"], 5)
            self.assertEqual(carga["total_steps"], 5)
            self.assertIn("límite", carga["motivo_guardado"])
            self.assertFalse(salida.with_name(salida.name + ".tmp").exists())

    def test_reanuda_sin_perder_calendario(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            salida = root / "ck.pt"
            base = _argumentos(root, salida)
            with contextlib.redirect_stdout(io.StringIO()):
                t.main(base + ["--pasos", "5", "--max-minutos", "0.0001"])
            primero = torch.load(salida, map_location="cpu", weights_only=True)
            with contextlib.redirect_stdout(io.StringIO()) as consola:
                codigo = t.main(base + ["--resume", str(salida), "--pasos", "1"])
            self.assertEqual(codigo, 0)
            carga = torch.load(salida, map_location="cpu", weights_only=True)
            self.assertEqual(carga["step"], primero["step"] + 1)
            self.assertEqual(carga["total_steps"], primero["total_steps"])
            self.assertIn("reanudado desde", consola.getvalue())

    def test_rechaza_checkpoint_de_otro_tokenizador(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            salida = root / "ck.pt"
            base = _argumentos(root, salida)
            with contextlib.redirect_stdout(io.StringIO()):
                t.main(base + ["--pasos", "1"])
            meta = json.loads((root / "datos" / "meta.json").read_text(encoding="utf-8"))
            meta["tokenizer_sha256"] = "f" * 64
            (root / "datos" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    t.main(base + ["--resume", str(salida), "--pasos", "1"])


if __name__ == "__main__":
    unittest.main()
