"""Comprobaciones deterministas del arnés de evaluación (sin GPU ni corpus)."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluar_modelo import cargar_tareas, comprobar, numero_final

PROMPTS = Path(__file__).resolve().parents[1] / "evals" / "prompts_es.jsonl"


class NumeroTests(unittest.TestCase):
    def test_ultimo_numero(self):
        self.assertEqual(numero_final("Primero 3, luego 12."), 12.0)

    def test_decimal_y_negativo(self):
        self.assertEqual(numero_final("resultado: -4,5"), -4.5)

    def test_sin_numero(self):
        self.assertIsNone(numero_final("no sé"))


class ReglaTests(unittest.TestCase):
    def test_numero(self):
        self.assertTrue(comprobar("Son 12 unidades", {"tipo": "numero", "valor": 12})[0])
        self.assertFalse(comprobar("Son 13 unidades", {"tipo": "numero", "valor": 12})[0])

    def test_contiene_todos(self):
        regla = {"tipo": "contiene", "valores": ["def", "return"]}
        self.assertTrue(comprobar("def f(): return 1", regla)[0])
        self.assertFalse(comprobar("def f(): pass", regla)[0])

    def test_contiene_alguno(self):
        regla = {"tipo": "contiene", "valores": ["float", "flotante"], "todos": False}
        self.assertTrue(comprobar("Es un FLOTANTE", regla)[0])
        self.assertFalse(comprobar("Es un entero", regla)[0])

    def test_min_palabras(self):
        regla = {"tipo": "min_palabras", "valor": 3}
        self.assertTrue(comprobar("una dos tres", regla)[0])
        self.assertFalse(comprobar("una dos", regla)[0])

    def test_no_vacio(self):
        self.assertFalse(comprobar("  ", {"tipo": "no_vacio"})[0])

    def test_regla_desconocida(self):
        with self.assertRaises(ValueError):
            comprobar("hola", {"tipo": "inventada"})


class TareasTests(unittest.TestCase):
    def test_prompts_reales(self):
        tareas = cargar_tareas(PROMPTS)
        self.assertEqual(len(tareas), 20)
        self.assertEqual(len({t["id"] for t in tareas}), 20)
        self.assertEqual({t["categoria"] for t in tareas},
                         {"espanol", "matematicas", "instrucciones", "programacion"})

    def test_linea_incompleta(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "mal.jsonl"
            ruta.write_text(json.dumps({"id": "x"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                cargar_tareas(ruta)

    def test_archivo_vacio(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "vacio.jsonl"
            ruta.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                cargar_tareas(ruta)


if __name__ == "__main__":
    unittest.main()
