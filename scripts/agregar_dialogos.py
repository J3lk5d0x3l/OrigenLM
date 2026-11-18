"""Agrega diálogos escritos a mano al corpus de entrenamiento.

Escribís un archivo de texto plano en `dataset/entradas/` y este script lo
convierte en pares `user`/`assistant` dentro de `dataset/chat_v2.jsonl`.

Formato aceptado (mayúsculas o minúsculas, con o sin `##`):

    ## PREGUNTA
    ¿Qué es un commit?

    ## RESPUESTA
    Es una foto del estado del repositorio en un momento dado.

También vale la forma corta:

    P: ¿Qué es un commit?
    R: Es una foto del estado del repositorio.

Después de agregar datos hay que regenerar el dataset de fine-tuning:

    python scripts/preparar_sft.py

Sin dependencias externas.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DESTINO = Path("dataset/chat_v2.jsonl")
ENTRADAS = Path("dataset/entradas")
IGNORAR = ("leeme", "ejemplo")

# El marcador tiene que ocupar la línea entera, o `MARCADOR: texto`.
def _marca(palabras: str) -> re.Pattern[str]:
    return re.compile(rf"^\s*(?:#{{1,6}}\s*)?({palabras})\s*(?::\s*(.*))?$", re.IGNORECASE)


PREGUNTA = _marca("pregunta|usuario|user|p")
RESPUESTA = _marca("respuesta|asistente|assistant|r")


def leer_pares(ruta: Path) -> list[tuple[str, str]]:
    """Extrae pares (pregunta, respuesta) de un archivo de texto."""
    pares: list[tuple[str, str]] = []
    usuario: list[str] | None = None
    asistente: list[str] | None = None
    modo = None

    def cerrar() -> None:
        if usuario is not None and asistente is not None:
            pregunta = " ".join(" ".join(usuario).split())
            respuesta = "\n".join(asistente).strip()
            if pregunta and respuesta:
                pares.append((pregunta, respuesta))

    for numero, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), 1):
        marca_p = PREGUNTA.match(linea)
        marca_r = RESPUESTA.match(linea)
        resto_p = marca_p.group(2) if marca_p else None
        resto_r = marca_r.group(2) if marca_r else None

        if marca_p:
            cerrar()
            usuario = [resto_p] if resto_p else []
            asistente = None
            modo = "p"
        elif marca_r:
            if usuario is None:
                print(f"[aviso] {ruta.name}:{numero}: respuesta sin pregunta, se ignora")
                modo = None
                continue
            asistente = [resto_r] if resto_r else []
            modo = "r"
        elif modo == "p" and usuario is not None:
            usuario.append(linea)
        elif modo == "r" and asistente is not None:
            asistente.append(linea)

    cerrar()
    return pares


def normalizar(texto: str) -> str:
    return " ".join(texto.split()).strip().lower()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entradas", type=Path, default=ENTRADAS,
                   help="carpeta con los .txt/.md escritos a mano")
    p.add_argument("--destino", type=Path, default=DESTINO)
    p.add_argument("--seco", action="store_true", help="Mostrar qué haría sin escribir")
    args = p.parse_args()

    if not args.entradas.is_dir():
        raise SystemExit(f"No existe la carpeta {args.entradas}")

    archivos = [f for f in sorted(args.entradas.iterdir())
                if f.suffix.lower() in (".txt", ".md")
                and not f.name.lower().startswith(IGNORAR)]
    if not archivos:
        raise SystemExit(
            f"No hay archivos para leer en {args.entradas}.\n"
            "Creá uno terminado en .txt y pegá los pares con el formato\n"
            "de LEEME.md (por ejemplo: dataset/entradas/mis_respuestas.txt)")

    existentes: list[dict] = []
    vistos: set[tuple[str, str]] = set()
    if args.destino.is_file():
        for numero, linea in enumerate(args.destino.read_text(encoding="utf-8").splitlines(), 1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                fila = json.loads(linea)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{args.destino}:{numero}: JSON inválido ({exc})")
            existentes.append(fila)
            vistos.add((normalizar(str(fila.get("user", ""))),
                        normalizar(str(fila.get("assistant", "")))))

    nuevos: list[tuple[str, str]] = []
    repetidos = 0
    for archivo in archivos:
        pares = leer_pares(archivo)
        print(f"  {archivo.name}: {len(pares)} par(es)")
        for pregunta, respuesta in pares:
            clave = (normalizar(pregunta), normalizar(respuesta))
            if clave in vistos:
                repetidos += 1
                continue
            vistos.add(clave)
            nuevos.append((pregunta, respuesta))

    if not nuevos:
        print(f"\nNada nuevo: {repetidos} repetido(s). Total sigue en {len(existentes)}.")
        return 0

    print(f"\nnuevos      : {len(nuevos)}")
    print(f"repetidos   : {repetidos}")
    print(f"total final : {len(existentes) + len(nuevos)}")
    for pregunta, _ in nuevos[:3]:
        print(f"  + {pregunta[:70]}")

    if args.seco:
        print("\n[seco] no se escribió nada.")
        return 0

    args.destino.parent.mkdir(parents=True, exist_ok=True)
    with args.destino.open("a", encoding="utf-8") as flujo:
        for pregunta, respuesta in nuevos:
            flujo.write(json.dumps({"user": pregunta, "assistant": respuesta},
                                   ensure_ascii=False) + "\n")
    print(f"\nescrito en {args.destino}")
    print("siguiente paso: python scripts/preparar_sft.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
