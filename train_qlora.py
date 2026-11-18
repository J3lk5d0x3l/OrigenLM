"""Fine-tuning (QLoRA) de un modelo abierto con los diálogos de OrigenLM.

Esto NO entrena un modelo desde cero: adapta un modelo abierto ya entrenado
(por defecto Qwen2.5-Coder-7B-Instruct, Apache 2.0) a los datos del repo.
Necesita GPU: en CPU un 7B es inviable. En Kaggle (T4 gratis) corre cómodo.

    python train_qlora.py --dry-run                     # valida datos y plan
    python train_qlora.py --epochs 1                    # entrena (GPU)
    python train_qlora.py --modelo Qwen/Qwen2.5-Coder-1.5B-Instruct --cpu --pasos 2

El adaptador queda en `checkpoints/qlora_origen/adapter/` y se puede cargar con
peft sobre el modelo base, o fusionar con `--fusionar` para exportar un modelo
completo.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

DEFAULT_MODELO = "Qwen/Qwen2.5-Coder-7B-Instruct"


def leer_jsonl(ruta: Path) -> list[dict]:
    if not ruta.is_file():
        raise SystemExit(f"No existe el archivo de datos: {ruta}")
    filas: list[dict] = []
    for numero, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), 1):
        linea = linea.strip()
        if not linea:
            continue
        try:
            fila = json.loads(linea)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{ruta}:{numero}: JSON inválido ({exc})")
        mensajes = fila.get("messages")
        if not isinstance(mensajes, list) or len(mensajes) < 2:
            raise SystemExit(f"{ruta}:{numero}: falta 'messages' con user/assistant")
        filas.append(fila)
    if not filas:
        raise SystemExit(f"{ruta} está vacío")
    return filas


def _ids(salida) -> list[int]:
    """Normaliza lo que devuelve apply_chat_template según la versión de transformers.

    En transformers 4 es una lista de enteros; en 5 es un BatchEncoding
    (o un tensor) y hay que sacar `input_ids` antes de comparar.
    """
    if hasattr(salida, "keys"):
        salida = salida["input_ids"]
    if hasattr(salida, "tolist"):
        salida = salida.tolist()
    if salida and isinstance(salida[0], (list, tuple)):
        salida = salida[0]
    return [int(x) for x in salida]


def construir_ejemplo(tokenizer, mensajes: list[dict], max_len: int):
    """Tokeniza y deja -100 en el prompt: solo se aprende de la respuesta."""
    prompt = _ids(tokenizer.apply_chat_template(mensajes[:-1], tokenize=True,
                                                add_generation_prompt=True))
    completo = _ids(tokenizer.apply_chat_template(mensajes, tokenize=True,
                                                 add_generation_prompt=False))

    if len(completo) > len(prompt) and completo[:len(prompt)] == prompt:
        ids, corte = completo, len(prompt)
    else:
        # Algunos tokenizadores no devuelven el prompt como prefijo exacto
        # (depende de la versión de transformers). Se arma a mano con el
        # texto de la respuesta y el token de fin.
        respuesta = _ids(tokenizer(mensajes[-1]["content"],
                                   add_special_tokens=False)["input_ids"])
        fin = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        ids, corte = prompt + respuesta + fin, len(prompt)

    ids = ids[:max_len]
    corte = min(corte, len(ids))
    etiquetas = [-100] * corte + ids[corte:]
    return ids, etiquetas


def agrupar(ejemplos, pad_id: int):
    largo = max(len(x) for x, _ in ejemplos)
    entrada, etiquetas, mascara = [], [], []
    for ids, labels in ejemplos:
        relleno = largo - len(ids)
        entrada.append(ids + [pad_id] * relleno)
        etiquetas.append(labels + [-100] * relleno)
        mascara.append([1] * len(ids) + [0] * relleno)
    return entrada, etiquetas, mascara


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", default=DEFAULT_MODELO)
    p.add_argument("--datos", type=Path, default=Path("dataset/sft/train.jsonl"))
    p.add_argument("--validacion", type=Path, default=Path("dataset/sft/val.jsonl"))
    p.add_argument("--salida", type=Path, default=Path("checkpoints/qlora_origen"))
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-acum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--calentamiento", type=int, default=10)
    p.add_argument("--eval-cada", type=int, default=0, help="0 = solo al final de cada época")
    p.add_argument("--guardar-cada", type=int, default=0, help="0 = solo al final de cada época")
    p.add_argument("--semilla", type=int, default=13)
    p.add_argument("--sin-4bit", action="store_true", help="Sin cuantización (más VRAM)")
    p.add_argument("--cpu", action="store_true", help="Forzar CPU (solo pruebas pequeñas)")
    p.add_argument("--pasos", type=int, default=0, help="Cortar tras N pasos (pruebas)")
    p.add_argument("--dry-run", action="store_true", help="Validar datos y plan sin cargar el modelo")
    p.add_argument("--fusionar", action="store_true", help="Fusionar adaptador y guardar modelo completo")
    args = p.parse_args()

    entrenamiento = leer_jsonl(args.datos)
    validacion = leer_jsonl(args.validacion) if args.validacion.is_file() else []
    caracteres = sum(len(m["content"]) for fila in entrenamiento for m in fila["messages"])
    pasos_por_epoca = math.ceil(len(entrenamiento) / args.batch)
    pasos = args.pasos or math.ceil(pasos_por_epoca * args.epochs)
    tokens_aprox = caracteres // 4

    print(f"modelo      : {args.modelo}")
    print(f"datos       : {len(entrenamiento)} ejemplos ({caracteres:,} caracteres, "
          f"~{tokens_aprox:,} tokens)")
    print(f"validación  : {len(validacion)} ejemplos")
    print(f"plan        : {pasos:,} pasos | batch {args.batch} x acumulación {args.grad_acum} "
          f"= {args.batch * args.grad_acum} por paso | contexto {args.max_len}")
    print(f"lora        : r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")
    print(f"salida      : {args.salida}")

    if tokens_aprox < 20_000:
        print("[aviso] menos de 20k tokens: alcanza para probar el flujo, "
              "no para enseñarle a conversar de verdad.")

    if args.dry_run:
        print("[dry-run] datos y plan válidos; no se cargó ningún modelo.")
        return 0

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.cpu and not torch.cuda.is_available():
        raise SystemExit(
            "No hay GPU. Este script está pensado para Kaggle/Colab.\n"
            "  - Para una prueba mínima: --cpu --modelo Qwen/Qwen2.5-Coder-1.5B-Instruct --pasos 2\n"
            "  - Para entrenar en serio: usá notebooks/kaggle_origenlm_qlora.ipynb")

    torch.manual_seed(args.semilla)
    dispositivo = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.modelo, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    carga = {"low_cpu_mem_usage": True}
    if dispositivo == "cuda" and not args.sin_4bit:
        from transformers import BitsAndBytesConfig
        carga["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        carga["device_map"] = "auto"
    else:
        carga["dtype"] = torch.float32 if dispositivo == "cpu" else torch.bfloat16

    modelo = AutoModelForCausalLM.from_pretrained(args.modelo, **carga)
    if dispositivo == "cuda" and not args.sin_4bit:
        modelo = prepare_model_for_kbit_training(modelo)
    modelo = get_peft_model(modelo, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    modelo.print_trainable_parameters()

    def preparar(filas):
        salida = []
        for fila in filas:
            ids, etiquetas = construir_ejemplo(tokenizer, fila["messages"], args.max_len)
            if any(e != -100 for e in etiquetas):
                salida.append((ids, etiquetas))
        return salida

    conjunto = preparar(entrenamiento)
    conjunto_val = preparar(validacion) if validacion else []
    if not conjunto:
        raise SystemExit("Ningún ejemplo aportó tokens de respuesta; revisá los datos.")

    optimizador = torch.optim.AdamW([q for q in modelo.parameters() if q.requires_grad],
                                    lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    calentamiento = max(1, min(args.calentamiento, pasos // 10 or 1))

    def tasa(paso: int) -> float:
        if paso < calentamiento:
            return args.lr * (paso + 1) / calentamiento
        avance = (paso - calentamiento) / max(1, pasos - calentamiento)
        return args.lr * 0.1 + (args.lr - args.lr * 0.1) * 0.5 * (1 + math.cos(math.pi * avance))

    def evaluar() -> float:
        if not conjunto_val:
            return float("nan")
        modelo.eval()
        perdidas = []
        with torch.no_grad():
            for inicio in range(0, len(conjunto_val), args.batch):
                x, y, m = agrupar(conjunto_val[inicio:inicio + args.batch], tokenizer.pad_token_id)
                lote = {"input_ids": torch.tensor(x, device=dispositivo),
                        "attention_mask": torch.tensor(m, device=dispositivo),
                        "labels": torch.tensor(y, device=dispositivo)}
                perdidas.append(float(modelo(**lote).loss))
        modelo.train()
        return sum(perdidas) / len(perdidas)

    args.salida.mkdir(parents=True, exist_ok=True)
    modelo.train()
    inicio = time.time()
    paso = 0
    perdida_acumulada = 0.0
    for epoca in range(math.ceil(args.epochs)):
        for comienzo in range(0, len(conjunto), args.batch):
            x, y, m = agrupar(conjunto[comienzo:comienzo + args.batch], tokenizer.pad_token_id)
            lote = {"input_ids": torch.tensor(x, device=dispositivo),
                    "attention_mask": torch.tensor(m, device=dispositivo),
                    "labels": torch.tensor(y, device=dispositivo)}
            perdida = modelo(**lote).loss / args.grad_acum
            perdida.backward()
            perdida_acumulada += float(perdida)
            if (paso + 1) % args.grad_acum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [q for q in modelo.parameters() if q.requires_grad], 1.0)
                for grupo in optimizador.param_groups:
                    grupo["lr"] = tasa(paso)
                optimizador.step()
                optimizador.zero_grad(set_to_none=True)
            paso += 1
            if paso % 10 == 0 or paso == pasos:
                transcurrido = time.time() - inicio
                print(f"paso {paso}/{pasos} | pérdida {perdida_acumulada / min(10, paso):.4f} "
                      f"| {paso / transcurrido:.2f} pasos/s", flush=True)
                perdida_acumulada = 0.0
            if args.eval_cada and paso % args.eval_cada == 0:
                print(f"  validación: pérdida {evaluar():.4f}", flush=True)
            if args.guardar_cada and paso % args.guardar_cada == 0:
                modelo.save_pretrained(args.salida / "adapter")
            if args.pasos and paso >= args.pasos:
                break
        if args.pasos and paso >= args.pasos:
            break

    perdida_val = evaluar()
    if args.fusionar:
        modelo = modelo.merge_and_unload()
        modelo.save_pretrained(args.salida / "modelo_fusionado")
        tokenizer.save_pretrained(args.salida / "modelo_fusionado")
    modelo.save_pretrained(args.salida / "adapter")
    tokenizer.save_pretrained(args.salida)
    (args.salida / "meta.json").write_text(json.dumps({
        "modelo_base": args.modelo,
        "ejemplos": len(conjunto),
        "pasos": paso,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_len": args.max_len,
        "perdida_validacion": perdida_val,
        "minutos": round((time.time() - inicio) / 60, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"listo: {args.salida}/adapter | pérdida validación "
          f"{'—' if math.isnan(perdida_val) else round(perdida_val, 4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
