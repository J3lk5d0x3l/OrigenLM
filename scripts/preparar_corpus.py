"""Preparar un corpus de archivos autorizados para entrenamiento en CPU."""
import argparse
from pathlib import Path
from origen.corpus import preparar_corpus


def main():
    p = argparse.ArgumentParser(description="Indexar documentos y tokenizar en disco")
    p.add_argument("--fuentes", type=Path, default=Path("dataset/fuentes"))
    p.add_argument("--salida", type=Path, default=Path("dataset/preparado"))
    p.add_argument("--max-mb", type=int, default=350)
    p.add_argument("--confirmar-derechos", action="store_true",
                   help="Confirmas tener derecho a usar las fuentes")
    args = p.parse_args()
    if not args.confirmar_derechos:
        p.error("Confirma los derechos de reutilización con --confirmar-derechos")
    data = preparar_corpus(args.fuentes, args.salida, args.max_mb * 1024 * 1024)
    print(f"Documentos: {data['total_documents']}")
    print(f"Tokens propios: {data['total_tokens']:,}")
    for k, item in data["splits"].items():
        print(f"{k}: {item['tokens']:,} tokens | {item['bytes'] / 1024 ** 2:.1f} MiB de disco")
    print(f"Archivo: {args.salida / 'meta.json'}")


if __name__ == "__main__":
    main()
