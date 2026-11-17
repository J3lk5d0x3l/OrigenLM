"""Cerrar la línea base V1 sin entrenar ni modificar sus checkpoints.

Guarda una copia local (NO una copia fuera de Codespaces), compara el NLL
de respuesta en el mismo conjunto reservado y genera pruebas cualitativas.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

import torch

from origen.chat_training import prepare_records, read_chat_dataset, response_nll
from origen.model import ModelConfig, OrigenLM
from origen.resources import configure_cpu_threads
from origen.tokenizer import DEFAULT_SYSTEM, conversation_prefix_limited, decode

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "inicial": ROOT / "checkpoints/origen_latest.pt",
    "masivo": ROOT / "checkpoints/origen_masivo.pt",
    "ajustado": ROOT / "checkpoints/origen_ajustado.pt",
}
PROMPTS = [
    ("saludo", "Hola, estoy aprendiendo programación. ¿Me saludas?", None),
    ("matematicas", "¿Cuánto es 4 + 2? Responde solo con el número.", "6"),
    ("matematicas", "Calcula 7 más 8. Responde solo con el número.", "15"),
    ("programacion", "¿Para qué sirve una función en Python?", None),
    ("seguridad", "¿Debería publicar mi contraseña en Internet?", None),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for part in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def backup_checkpoint_files(files: dict[str, Path], out: Path) -> dict:
    """ZIP local con pesos y manifiesto; sin datasets ni secretos."""
    existing = {name: path for name, path in files.items() if path.is_file()}
    if not existing:
        raise FileNotFoundError("No hay checkpoints V1 para respaldar.")
    manifest = {
        "format": "origen-v1-backup-1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {},
        "warning": "ZIP local del Codespace: descargalo o guardalo externamente.",
    }
    for name, path in existing.items():
        manifest["files"][name] = {
            "filename": path.name, "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for name, path in existing.items():
                archive.write(path, arcname=path.name)
        with zipfile.ZipFile(tmp, "r") as archive:
            if archive.testzip() is not None:
                raise OSError("No fue posible verificar el ZIP.")
            for entry in manifest["files"].values():
                if hashlib.sha256(archive.read(entry["filename"])).hexdigest() != entry["sha256"]:
                    raise OSError("Checksum incorrecto en la copia del checkpoint.")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return manifest


def deterministic_samples(model: OrigenLM, max_new_tokens: int = 50) -> list[dict]:
    results = []
    for subject, question, expected in PROMPTS:
        prompt = conversation_prefix_limited(DEFAULT_SYSTEM, [], question, model.config.max_seq_len)
        # top_k=1: decodificación determinista sin muestreo aleatorio.
        tokens = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=1.0, top_k=1)
        reply = decode(tokens).strip()
        exact = None
        if expected is not None:
            exact = bool(re.fullmatch(re.escape(expected) + r"[.!]?", reply))
        results.append({"materia": subject, "pregunta": question,
                        "respuesta": reply, "esperada": expected, "correcta": exact})
    return results


def load_v1(path: Path, context: int):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("architecture") not in (None, "origen-v1"):
        raise ValueError(f"No es checkpoint v1: {path}")
    cfg = ModelConfig(**saved["config"])
    cfg.max_seq_len = max(cfg.max_seq_len, context)
    model = OrigenLM(cfg)
    model.load_state_dict(saved["model"])
    model.eval()
    return saved, model


def inspect_data(train_path: Path, val_path: Path):
    train = read_chat_dataset(train_path)
    validation = read_chat_dataset(val_path)
    train_questions = {item["user"].strip().casefold() for item in train}
    val_questions = {item["user"].strip().casefold() for item in validation}
    overlap = sorted(train_questions & val_questions)
    return {
        "entrenamiento": len(train), "validacion": len(validation),
        "solapamiento_exacto": len(overlap),
        "solapamiento_muestra": overlap[:4],
        "advertencia": "La igualdad de preguntas no detecta paráfrasis ni contaminación del corpus.",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Respaldar y evaluar OrigenLM V1 antes de pasar a V2")
    p.add_argument("--output-dir", type=Path, default=ROOT / "dist/v1")
    p.add_argument("--context", type=int, default=384)
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--no-backup", action="store_true", help="Solo pruebas; no crear ZIP")
    args = p.parse_args(argv)
    if min(args.context, args.max_new_tokens) < 1:
        p.error("context y max-new-tokens deben ser positivos")
    if args.context > 2048:
        p.error("contexto de evaluación máximo 2048 para esta herramienta")
    configure_cpu_threads("chat")
    originals = {name: path for name, path in CHECKPOINTS.items() if path.is_file()}
    if not originals:
        p.error("No hay checkpoints v1. Revisa origen estado.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    print("[OrigenLM] Cerrando la línea base V1. No se entrenan ni sobrescriben pesos.", flush=True)
    backup = None
    if not args.no_backup:
        backup = output / f"origen_v1_respaldo_{stamp}.zip"
        manifest = backup_checkpoint_files(originals, backup)
        print(f"[OrigenLM] ZIP local verificado: {backup} ({len(manifest['files'])} checkpoints)", flush=True)
    evaluation = read_chat_dataset(ROOT / "dataset/evaluacion_v2.jsonl")
    dataset = inspect_data(ROOT / "dataset/chat_v2.jsonl", ROOT / "dataset/evaluacion_v2.jsonl")
    if dataset["solapamiento_exacto"]:
        p.error("Hay preguntas iguales en datos de entrenamiento y evaluación.")
    results = {}
    for name, path in originals.items():
        try:
            checkpoint, model = load_v1(path, args.context)
            samples = prepare_records(evaluation, model.config.max_seq_len)
            nll, supervised_tokens = response_nll(model, samples)
            result = {
                "checkpoint": str(path.relative_to(ROOT)), "step": int(checkpoint.get("step", 0)),
                "parameters": model.count_parameters(), "context_evaluado": model.config.max_seq_len,
                "response_nll": float(nll), "examples": len(evaluation),
                "supervised_tokens": supervised_tokens, "sha256": sha256_file(path),
            }
            if name == "ajustado":
                result["generaciones"] = deterministic_samples(model, args.max_new_tokens)
            results[name] = result
            print(f"[{name}] paso={result['step']} | NLL respuestas={nll:.4f} | "
                  f"{supervised_tokens} tokens | {result['parameters']:,} parámetros", flush=True)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            print(f"[{name}] No fue posible evaluarlo: {exc}", flush=True)
            results[name] = {"error": str(exc)}
    report = {
        "format": "origen-v1-baseline-1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset, "resultados": results,
        "respaldo_local": str(backup) if backup else None,
        "nota": ("Evaluación limitada a 15 ejemplos de demostración; NLL no demuestra comprensión. "
                 "Ampliar RoPE no equivale a aprender contexto largo. "
                 "Este respaldo no sobrevive si se elimina el Codespace."),
    }
    path = output / f"evaluacion_v1_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OrigenLM] Informe: {path}", flush=True)
    if "generaciones" in results.get("ajustado", {}):
        correct = sum(x["correcta"] is True for x in results["ajustado"]["generaciones"])
        math_tests = sum(x["esperada"] is not None for x in results["ajustado"]["generaciones"])
        print(f"[OrigenLM] Aritmética generada: {correct}/{math_tests} respuestas exactas "
              "(miniprueba, no benchmark general).", flush=True)
        for item in results["ajustado"]["generaciones"]:
            print(f"  [{item['materia']}] {item['pregunta']}\n       -> {item['respuesta'][:180]}", flush=True)
    if backup:
        print("[IMPORTANTE] Descarga o exporta el ZIP fuera del Codespace; "
              "un respaldo local no protege contra su eliminación.", flush=True)
    print("[SIGUIENTE] Conservar V1 y diseñar el preentrenamiento de V2 en un corpus ampliado.", flush=True)
    return 0 if all("error" not in r for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
