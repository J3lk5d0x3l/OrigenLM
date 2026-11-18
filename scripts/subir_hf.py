"""Sube o baja el adaptador QLoRA de OrigenLM en el Hugging Face Hub.

Subir (en Kaggle, al terminar el entrenamiento):
    python scripts/subir_hf.py --repo origenlm-qlora --carpeta /kaggle/working/qlora_origen

Bajar (en el codespace o en tu PC):
    python scripts/subir_hf.py --bajar --repo origenlm-qlora --carpeta checkpoints/qlora_origen

El token se lee de la variable HF_TOKEN (con permiso Write para subir). Si `--repo`
no lleva usuario ("origenlm-qlora"), se completa con el dueño del token. El repo se
crea privado salvo que pases `--publico`.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def obtener_token(obligatorio: bool) -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if obligatorio and not token:
        raise SystemExit(
            "Falta HF_TOKEN. Crea un token con permiso Write en "
            "https://huggingface.co/settings/tokens y exportalo:\n"
            "  export HF_TOKEN=hf_...\n"
            "En Kaggle: Add-ons > Secrets > HF_TOKEN.")
    return token


def nombre_completo(api, repo: str) -> str:
    return repo if "/" in repo else f"{api.whoami()['name']}/{repo}"


def subir(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    carpeta: Path = args.carpeta
    if not (carpeta / "adapter" / "adapter_config.json").is_file():
        raise SystemExit(
            f"No encuentro {carpeta}/adapter/adapter_config.json: "
            "¿terminó el entrenamiento y es la carpeta correcta?")

    api = HfApi(token=obtener_token(True))
    repo = nombre_completo(api, args.repo)
    api.create_repo(repo, repo_type="model", private=not args.publico, exist_ok=True)

    ignorar = ["checkpoint-*/**"]
    if not args.con_fusionado:
        ignorar.append("modelo_fusionado/**")
    api.upload_folder(folder_path=str(carpeta), repo_id=repo, repo_type="model",
                      ignore_patterns=ignorar, commit_message=args.mensaje)
    print(f"listo: https://huggingface.co/{repo} "
          f"({'público' if args.publico else 'privado'})")


def bajar(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi, snapshot_download

    token = obtener_token(False)
    repo = args.repo
    if "/" not in repo:
        if not token:
            raise SystemExit("Pasa --repo usuario/nombre o define HF_TOKEN.")
        repo = nombre_completo(HfApi(token=token), repo)
    destino = snapshot_download(repo_id=repo, repo_type="model",
                                local_dir=str(args.carpeta), token=token)
    print(f"listo: {destino}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="origenlm-qlora",
                   help="nombre del repo en Hugging Face (con o sin usuario)")
    p.add_argument("--carpeta", type=Path, default=Path("checkpoints/qlora_origen"),
                   help="carpeta local del entrenamiento (la de --salida)")
    p.add_argument("--bajar", action="store_true", help="descargar en vez de subir")
    p.add_argument("--publico", action="store_true", help="crear el repo público")
    p.add_argument("--con-fusionado", action="store_true",
                   help="incluir modelo_fusionado/ (pesa varios GB)")
    p.add_argument("--mensaje", default="Adaptador QLoRA de OrigenLM")
    args = p.parse_args()

    if args.bajar:
        bajar(args)
    else:
        subir(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
