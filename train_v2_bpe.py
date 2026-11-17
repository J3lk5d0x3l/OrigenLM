"""Preentrenamiento propio de OrigenLM V2 (BPE) sobre bins uint16, CPU o GPU.

Diseño de reanudación: el calendario (warmup + coseno) se calcula con el paso
ABSOLUTO y un total absoluto guardado en el checkpoint, así que `--resume`
continúa la curva donde quedó en vez de reiniciar el warmup. Todo corte
(límite de tiempo, SIGINT/SIGTERM) escribe un checkpoint válido antes de salir.

No descarga ni usa pesos de terceros: solo lee los bins y el vocabulario
propios indicados por argumentos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from origen.resources import configure_cpu_threads
from origen.v2 import OrigenV2, OrigenV2Config

ARQUITECTURA = "origen-v2-gqa-v1"
FORMATO_DATOS = "origen-bpe-v2"


# --------------------------------------------------------------------------
# Lógica pura (sin E/S, cubierta por tests/test_train_v2_bpe.py)
# --------------------------------------------------------------------------

def tasa_aprendizaje(paso: int, total: int, warmup: int, lr: float,
                     min_ratio: float = 0.1) -> float:
    """LR del paso ABSOLUTO `paso` dentro de un calendario de `total` pasos.

    Reanudar no reinicia el warmup: en el paso 460 de 3000 la tasa es la de la
    curva original, no `lr/warmup`.
    """
    if paso < 1:
        raise ValueError("El paso debe ser >= 1")
    if total < 1 or warmup < 0 or lr <= 0 or not 0 <= min_ratio <= 1:
        raise ValueError("Parámetros de calendario inválidos")
    if paso <= warmup:
        return lr * paso / max(1, warmup)
    avance = (paso - warmup) / max(1, total - warmup)
    avance = min(1.0, max(0.0, avance))
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * avance)))


def planificar_pasos(desde: int, pasos: int, total_explicito: int | None = None,
                     total_guardado: int | None = None) -> tuple[int, int]:
    """Devuelve (paso final de esta ejecución, total del calendario del coseno).

    - `--pasos` son los pasos NUEVOS a ejecutar ahora.
    - `--pasos-total` es el horizonte del coseno: no obliga a entrenar hasta él.
    - Al reanudar se conserva el total guardado si es mayor, para no acortar
      la curva; si el checkpoint es antiguo y no lo trae, hay que indicarlo.
    """
    if desde < 0 or pasos < 1:
        raise ValueError("El número de pasos debe ser positivo")
    solicitado = desde + pasos
    if total_explicito is not None:
        if total_explicito <= desde:
            raise ValueError(
                f"--pasos-total ({total_explicito}) debe superar el paso guardado ({desde})")
        return min(solicitado, total_explicito), total_explicito
    return solicitado, max(solicitado, total_guardado or 0)


def huella_sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with Path(ruta).open("rb") as flujo:
        for bloque in iter(lambda: flujo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def huella_opcional(ruta: Path) -> str | None:
    ruta = Path(ruta)
    return huella_sha256(ruta) if ruta.is_file() else None


def validar_compatibilidad(carga: dict, meta: dict, tokenizer_sha256: str | None,
                           forzar: bool = False) -> list[str]:
    """Comprueba que checkpoint, bins y tokenizador hablen el mismo idioma.

    Devuelve avisos. Lanza ValueError si hay incompatibilidad real, salvo
    `forzar=True`, que la degrada a aviso (para experimentos conscientes).
    """
    problemas: list[str] = []
    if meta.get("format") != FORMATO_DATOS:
        problemas.append(
            f"el corpus no tiene formato {FORMATO_DATOS} (tiene {meta.get('format')!r})")
    config = carga.get("config") or {}
    vocab_ckpt = int(config.get("vocab_size") or 0)
    vocab_meta = int(meta.get("vocab_size") or 0)
    if vocab_ckpt and vocab_meta and vocab_ckpt != vocab_meta:
        problemas.append(
            f"el checkpoint usa vocab {vocab_ckpt} y el corpus {vocab_meta}")
    sha_ckpt = carga.get("tokenizer_sha256")
    sha_meta = meta.get("tokenizer_sha256")
    if sha_ckpt and sha_meta and sha_ckpt != sha_meta:
        problemas.append("el checkpoint se entrenó con otro tokenizer.json")
    if tokenizer_sha256 and sha_ckpt and tokenizer_sha256 != sha_ckpt:
        problemas.append("el tokenizer.json indicado no es el del checkpoint")
    if problemas and not forzar:
        raise ValueError("Checkpoint incompatible con el corpus:\n- "
                         + "\n- ".join(problemas)
                         + "\nUsa los archivos originales o --forzar-compatibilidad.")
    return [f"forzado: {p}" for p in problemas]


@dataclass(frozen=True)
class Precision:
    """Precisión efectiva para el paso hacia adelante del entrenamiento."""

    dtype: torch.dtype | None
    necesita_scaler: bool

    @property
    def nombre(self) -> str:
        if self.dtype is None:
            return "fp32"
        return {torch.bfloat16: "bf16", torch.float16: "fp16"}[self.dtype]


def resolver_dispositivo(pedido: str) -> torch.device:
    pedido = (pedido or "auto").lower()
    if pedido == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if pedido == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "Pediste --device cuda pero este torch no ve GPU. "
            "Instala una rueda CUDA o usa --device cpu / --device auto.")
    if pedido not in {"cpu", "cuda"}:
        raise SystemExit(f"Dispositivo no soportado: {pedido}")
    return torch.device(pedido)


def resolver_precision(dispositivo: torch.device, pedido: str = "auto") -> Precision:
    """bf16 si el hardware lo soporta; si no, fp16 con GradScaler; si no, fp32."""
    pedido = (pedido or "auto").lower()
    if pedido not in {"auto", "fp32", "bf16", "fp16"}:
        raise SystemExit(f"Precisión no soportada: {pedido}")
    if pedido == "fp32" or dispositivo.type == "cpu":
        return Precision(None, False)
    if pedido == "auto" and dispositivo.type == "cuda" and not torch.cuda.is_bf16_supported():
        pedido = "fp16"
    elif pedido == "auto":
        pedido = "bf16"
    if pedido == "bf16":
        return Precision(torch.bfloat16, False)
    return Precision(torch.float16, True)


def a_cpu(objeto):
    """Copia tensores a CPU para que el checkpoint sea portable CPU/GPU."""
    if torch.is_tensor(objeto):
        return objeto.detach().to("cpu")
    if isinstance(objeto, dict):
        return {clave: a_cpu(valor) for clave, valor in objeto.items()}
    if isinstance(objeto, (list, tuple)):
        return type(objeto)(a_cpu(valor) for valor in objeto)
    return objeto


def guardar_checkpoint(ruta: Path, carga: dict) -> Path:
    """Escritura atómica: nunca deja un checkpoint a medias en `ruta`."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".tmp")
    torch.save(a_cpu(carga), temporal)
    try:
        with temporal.open("rb") as flujo:
            os.fsync(flujo.fileno())
    except OSError:
        pass
    os.replace(temporal, ruta)
    return ruta


def crear_scaler(dispositivo: torch.device, habilitado: bool):
    """GradScaler compatible con torch >= 2.2 (API nueva y antigua)."""
    try:
        return torch.amp.GradScaler(dispositivo.type, enabled=habilitado)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=habilitado)


def cargar_bins(carpeta: Path):
    """Abre el corpus. `train` puede tener varios shards; `val` es uno solo.

    Formato de un shard: `train-000.bin`, `train-001.bin`, … listados en
    `meta["splits"]["train"]["paths"]`. Con un solo archivo se usa `path`.
    """
    carpeta = Path(carpeta)
    meta = json.loads((carpeta / "meta.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMATO_DATOS:
        raise ValueError(f"{carpeta}/meta.json no tiene formato {FORMATO_DATOS}")
    train = meta["splits"]["train"]
    rutas = [carpeta / r for r in train.get("paths", [train["path"]])]
    faltan = [str(r) for r in rutas if not r.is_file()]
    if faltan:
        raise ValueError(f"Faltan shards de entrenamiento: {faltan}")
    val_meta = meta["splits"]["val"]
    if "paths" in val_meta:
        raise ValueError("La validación debe ser un único .bin para que el NLL "
                         "sea comparable entre pasos")
    val = np.memmap(carpeta / val_meta["path"], dtype="<u2", mode="r")
    return meta, {"train": Ventanas(rutas), "val": val}


class Ventanas:
    """Ventanas aleatorias sobre uno o varios shards uint16 de entrenamiento.

    Cada fila del lote elige shard con probabilidad proporcional a sus tokens,
    así añadir corpus es concatenar bins (`paths` en meta.json) sin copiar nada.
    """

    def __init__(self, rutas):
        self.bins = [np.memmap(Path(ruta), dtype="<u2", mode="r") for ruta in rutas]
        self.tokens = sum(len(bin_) for bin_ in self.bins)
        self.pesos = np.array([len(bin_) for bin_ in self.bins], dtype=np.float64)
        self.pesos /= self.tokens
        self.shards = len(self.bins)

    def lote(self, ctx: int, batch: int, gen: np.random.Generator):
        if self.tokens <= ctx + 2:
            raise ValueError("El split es más corto que el contexto pedido")
        if self.shards == 1:
            elegidos = np.zeros(batch, dtype=np.int64)
        else:
            elegidos = gen.choice(self.shards, size=batch, p=self.pesos)
        entradas = np.empty((batch, ctx), dtype=np.int64)
        objetivos = np.empty((batch, ctx), dtype=np.int64)
        for fila, indice in enumerate(elegidos):
            flujo = self.bins[int(indice)]
            limite = len(flujo) - ctx - 2
            if limite <= 1:
                raise ValueError(f"El shard {indice} es más corto que el contexto")
            inicio = int(gen.integers(0, limite))
            entradas[fila] = flujo[inicio:inicio + ctx]
            objetivos[fila] = flujo[inicio + 1:inicio + 1 + ctx]
        return torch.from_numpy(entradas), torch.from_numpy(objetivos)


def muestrear(flujo, ctx: int, batch: int, gen: np.random.Generator):
    """Compatibilidad: ventanas de un único flujo abierto (memmap o array)."""
    limite = len(flujo) - ctx - 2
    if limite <= 1:
        raise ValueError("El split es más corto que el contexto pedido")
    inicios = gen.integers(0, limite, size=batch)
    entradas = np.empty((batch, ctx), dtype=np.int64)
    objetivos = np.empty((batch, ctx), dtype=np.int64)
    for fila, inicio in enumerate(inicios):
        entradas[fila] = flujo[inicio:inicio + ctx]
        objetivos[fila] = flujo[inicio + 1:inicio + 1 + ctx]
    return torch.from_numpy(entradas), torch.from_numpy(objetivos)


def sincronizar(dispositivo: torch.device) -> None:
    if dispositivo.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluar(modelo, flujo, ctx: int, batch: int, dispositivo: torch.device,
            max_tokens: int = 200_000):
    """NLL medio por token en un tramo contiguo y fijo (comparable entre pasos).

    Se usa no_grad (no inference_mode): las tablas de RoPE se guardan en caché
    y unos tensores de inferencia reutilizados en entrenamiento rompen autograd.
    La evaluación va en fp32 incluso con AMP para que el NLL sea comparable.
    """
    estaba_entrenando = modelo.training
    modelo.eval()
    total = 0.0
    vistos = 0
    inicio = 0
    try:
        while vistos < max_tokens and inicio + batch * ctx + 1 <= len(flujo):
            x = np.empty((batch, ctx), dtype=np.int64)
            y = np.empty((batch, ctx), dtype=np.int64)
            for fila in range(batch):
                desde = inicio + fila * ctx
                x[fila] = flujo[desde:desde + ctx]
                y[fila] = flujo[desde + 1:desde + 1 + ctx]
            xt = torch.from_numpy(x).to(dispositivo)
            yt = torch.from_numpy(y).to(dispositivo)
            with torch.autocast(device_type=dispositivo.type, enabled=False):
                _, perdida = modelo(xt, yt)
            total += float(perdida) * xt.numel()
            vistos += xt.numel()
            inicio += batch * ctx
    finally:
        modelo.train(estaba_entrenando)
    if not vistos:
        raise ValueError("Validación demasiado corta para el contexto pedido")
    return total / vistos, vistos


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preentrenar OrigenLM V2 (BPE propio) en CPU o GPU")
    p.add_argument("--datos", type=Path, default=Path("dataset/preparado_v2_es"))
    p.add_argument("--config", type=Path, default=Path("configs/origen_es_7m.json"))
    p.add_argument("--salida", type=Path, default=Path("checkpoints/origen_v2_es.pt"))
    p.add_argument("--pasos", type=int, default=20000,
                   help="Pasos nuevos (o total si se empieza desde cero)")
    p.add_argument("--pasos-total", type=int,
                   help="Paso final absoluto del calendario (manda sobre --pasos)")
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--guardar-cada", type=int, default=500)
    p.add_argument("--eval-cada", type=int, default=250)
    p.add_argument("--eval-max-tokens", type=int, default=200_000)
    p.add_argument("--max-minutos", type=float, default=0.0, help="0 = sin límite")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--precision", default="auto",
                   choices=("auto", "fp32", "bf16", "fp16"))
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--resume", type=Path)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--forzar-compatibilidad", action="store_true",
                   help="Acepta un checkpoint de otro tokenizador (bajo tu riesgo)")
    return p


def main(argv: list[str] | None = None) -> int:
    p = construir_parser()
    args = p.parse_args(argv)

    if min(args.pasos, args.ctx, args.batch_size, args.grad_accum,
           args.guardar_cada, args.eval_cada) < 1 or args.lr <= 0:
        p.error("pasos, ctx, batch, acumulación, guardado, eval y lr deben ser positivos")
    if args.salida.exists() and not (args.overwrite or args.resume):
        p.error("El checkpoint ya existe: usa --resume o --overwrite")

    meta, datos = cargar_bins(args.datos)
    train, val = datos["train"], datos["val"]
    dispositivo = resolver_dispositivo(args.device)
    precision = resolver_precision(dispositivo, args.precision)
    if dispositivo.type == "cpu":
        hilos = configure_cpu_threads("train")
    else:
        hilos = 0
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    previo = None
    if args.resume:
        previo = torch.load(args.resume, map_location="cpu", weights_only=True)
        if previo.get("architecture") != ARQUITECTURA:
            p.error(f"El checkpoint no es {ARQUITECTURA}")
        avisos = []
        try:
            avisos = validar_compatibilidad(
                previo, meta,
                huella_opcional(args.datos / meta.get("tokenizer", "tokenizer.json")),
                args.forzar_compatibilidad)
        except ValueError as exc:
            p.error(str(exc))
        for aviso in avisos:
            print(f"[aviso] {aviso}")
        cfg = OrigenV2Config(**previo["config"])
        inicio_paso = int(previo["step"])
        tokens_previos = int(previo.get("tokens_vistos", 0))
        ultima_val = previo.get("val_loss")
        paso_val = previo.get("val_loss_step", inicio_paso)
    else:
        cfg = OrigenV2Config(**json.loads(args.config.read_text(encoding="utf-8")))
        inicio_paso = 0
        tokens_previos = 0
        ultima_val = None
        paso_val = 0
    if args.ctx > cfg.max_seq_len:
        p.error(f"--ctx {args.ctx} supera max_seq_len {cfg.max_seq_len}")
    cfg.validate()

    try:
        fin, total = planificar_pasos(inicio_paso, args.pasos, args.pasos_total,
                                      (previo or {}).get("total_steps"))
    except ValueError as exc:
        p.error(str(exc))

    torch.manual_seed(args.seed)
    modelo = OrigenV2(cfg)
    if previo:
        modelo.load_state_dict(previo["model"])
    modelo.to(dispositivo)
    decay, sin_decay = [], []
    for nombre, parametro in modelo.named_parameters():
        (sin_decay if parametro.ndim < 2 or "tokens" in nombre else decay).append(parametro)
    optimizador = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": sin_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=False)
    if previo and "optimizer" in previo:
        optimizador.load_state_dict(previo["optimizer"])
    scaler = crear_scaler(dispositivo, precision.necesita_scaler)

    batch_efectivo = args.batch_size * args.grad_accum
    tokens_paso = batch_efectivo * args.ctx
    print(f"OrigenLM V2 | {modelo.count_parameters():,} parámetros | "
          f"dispositivo={dispositivo.type} | precisión={precision.nombre} | hilos={hilos}")
    print(f"datos: {meta['total_tokens']:,} tokens | vocab={meta['vocab_size']} | "
          f"ctx={args.ctx} | batch efectivo={batch_efectivo} "
          f"({tokens_paso:,} tokens/paso)")
    print(f"ejecución: pasos {inicio_paso:,} → {fin:,} "
          f"({fin - inicio_paso:,} nuevos, "
          f"{(fin - inicio_paso) * tokens_paso / 1e6:.1f}M tokens) | "
          f"calendario del coseno hasta {total:,}")
    if inicio_paso and args.pasos_total is None and not (previo or {}).get("total_steps"):
        print("[aviso] El checkpoint no guardó el horizonte del coseno: "
              "usa --pasos-total para conservar la curva de LR.")
    if previo:
        print(f"reanudado desde {args.resume} | tokens previos {tokens_previos:,} | "
              f"val {ultima_val if ultima_val is None else round(float(ultima_val), 4)}")
    if dispositivo.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if ultima_val is None:
        ultima_val, tokens_val = evaluar(modelo, val, args.ctx, args.batch_size,
                                         dispositivo, args.eval_max_tokens)
        paso_val = inicio_paso
        print(f"validación inicial: NLL={ultima_val:.4f} ({tokens_val:,} tokens)", flush=True)

    class Corte:
        pedido = False
        motivo = ""

    def _senal(numero, _marco):
        Corte.pedido = True
        Corte.motivo = f"señal {signal.Signals(numero).name}"
        print(f"\n[aviso] {Corte.motivo} recibida: se guardará al terminar este paso.",
              flush=True)

    anteriores = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            anteriores[sig] = signal.signal(sig, _senal)
        except (ValueError, OSError):
            pass

    def guardar(paso: int, motivo: str | None) -> None:
        carga = {
            "architecture": ARQUITECTURA,
            "config": cfg.__dict__.copy(),
            "step": paso,
            "total_steps": total,
            "run_end": fin,
            "model": modelo.state_dict(),
            "optimizer": optimizador.state_dict(),
            "train_loss": perdida_total,
            "val_loss": ultima_val,
            "val_loss_step": paso_val,
            "tokens_vistos": tokens_previos + tokens_curso,
            "tokens_paso": tokens_paso,
            "data": str(args.datos),
            "tokenizer_sha256": meta.get("tokenizer_sha256"),
            "device": dispositivo.type,
            "precision": precision.nombre,
            "motivo_guardado": motivo or "programado",
            "minutos": (time.time() - arranque) / 60,
        }
        guardar_checkpoint(args.salida, carga)
        print(f"  guardado: {args.salida} | paso {paso} | "
              f"motivo={carga['motivo_guardado']}", flush=True)

    gen = np.random.default_rng(args.seed + inicio_paso)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    tokens_curso = 0
    arranque = time.time()
    marca = time.time()
    sincronizar(dispositivo)
    pasos_medidos = 0
    perdida_total = 0.0
    motivo = None
    paso = inicio_paso

    try:
        for paso in range(inicio_paso + 1, fin + 1):
            lr = tasa_aprendizaje(paso, total, args.warmup, args.lr, args.min_lr_ratio)
            for grupo in optimizador.param_groups:
                grupo["lr"] = lr
            modelo.train()
            optimizador.zero_grad(set_to_none=True)
            perdida_total = 0.0
            for _ in range(args.grad_accum):
                x, y = train.lote(args.ctx, args.batch_size, gen)
                if dispositivo.type == "cuda":
                    x = x.pin_memory()
                    y = y.pin_memory()
                x = x.to(dispositivo, non_blocking=True)
                y = y.to(dispositivo, non_blocking=True)
                with torch.autocast(device_type=dispositivo.type,
                                    dtype=precision.dtype,
                                    enabled=precision.dtype is not None):
                    _, perdida = modelo(x, y)
                escalado = perdida / args.grad_accum
                if scaler.is_enabled():
                    scaler.scale(escalado).backward()
                else:
                    escalado.backward()
                perdida_total += float(perdida.detach()) / args.grad_accum
            if scaler.is_enabled():
                scaler.unscale_(optimizador)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            if scaler.is_enabled():
                scaler.step(optimizador)
                scaler.update()
            else:
                optimizador.step()
            tokens_curso += tokens_paso
            pasos_medidos += 1

            if paso % 20 == 0 or paso == inicio_paso + 1:
                sincronizar(dispositivo)
                segundos = max(1e-6, time.time() - marca)
                ritmo = tokens_paso * pasos_medidos / segundos
                pasos_medidos = 0
                marca = time.time()
                restante = (fin - paso) * tokens_paso
                eta = restante / ritmo / 60 if ritmo else 0.0
                print(f"paso {paso:6d} | NLL={perdida_total:.4f} | lr={lr:.2e} | "
                      f"{ritmo:,.0f} tok/s | ETA {eta:.0f} min", flush=True)

            motivo = None
            if Corte.pedido:
                motivo = Corte.motivo
            elif args.max_minutos and (time.time() - arranque) / 60 >= args.max_minutos:
                motivo = f"límite de {args.max_minutos:g} min"
            if motivo:
                print(f"{motivo} en el paso {paso}: se guarda antes de salir.", flush=True)

            if (paso % args.eval_cada == 0 or paso == fin) and not motivo:
                ultima_val, tokens_val = evaluar(
                    modelo, val, args.ctx, args.batch_size, dispositivo,
                    args.eval_max_tokens)
                paso_val = paso
                print(f"  validación: NLL={ultima_val:.4f} ({tokens_val:,} tokens) | "
                      f"perplejidad={math.exp(min(20, ultima_val)):.2f}", flush=True)

            if paso % args.guardar_cada == 0 or paso == fin or motivo:
                guardar(paso, motivo)
            if motivo:
                break
    finally:
        for sig, manejador in anteriores.items():
            try:
                signal.signal(sig, manejador)
            except (ValueError, OSError):
                pass

    sincronizar(dispositivo)
    minutos = (time.time() - arranque) / 60
    ritmo_medio = tokens_curso / max(1e-6, minutos * 60)
    print(f"RESUMEN | pasos nuevos={paso - inicio_paso} | "
          f"tokens nuevos={tokens_curso:,} | {minutos:.1f} min | "
          f"{ritmo_medio:,.0f} tok/s medios (incluye evaluación)")
    if ultima_val is not None:
        print(f"RESUMEN | NLL val={float(ultima_val):.4f} (paso {paso_val}) | "
              f"perplejidad={math.exp(min(20, float(ultima_val))):.2f}")
    if dispositivo.type == "cuda":
        print(f"RESUMEN | pico de memoria GPU="
              f"{torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB")
    print("Terminado. Compara respuestas reales antes de asumir mejora.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
