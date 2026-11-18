# Cómo entrenar y usar OrigenLM (estado actual)

Este documento reemplaza a las guías de la etapa anterior. Hay **tres caminos
distintos** y conviene no mezclarlos.

| Camino | Qué hace | Dónde corre | Velocidad real |
|---|---|---|---|
| 1. Chatear con un modelo abierto | Usa Qwen2.5-Coder-7B ya entrenado | Codespace (CPU) | ~6 tok/s |
| 2. Fine-tuning (QLoRA) de un modelo abierto | Adapta Qwen a tus datos | GPU (Kaggle T4) | 1 época en minutos |
| 3. Entrenar el modelo propio v2 | Preentrena `origen/v2.py` desde cero | GPU (Kaggle T4) | días para algo usable |

> **El codespace no entrena.** Tiene 4 vCPU y 16 GB: sirve para programar,
> preparar datos y conversar. El entrenamiento va en GPU gratuita (Kaggle).

## 1. Chatear ahora (sin entrenar nada)

```bash
bash scripts/chat_local.sh            # Qwen2.5-Coder-7B (recomendado)
MODEL=models/Qwen3-8B-Q4_K_M.gguf bash scripts/chat_local.sh
bash scripts/chat_local.sh --max-new-tokens 800
```

La primera vez descarga ~4,4 GB en `models/` (fuera de Git). El servidor queda
en `http://127.0.0.1:8080/v1` y habla la API de OpenAI, así que también sirve
para conectar extensiones de VS Code (Cline/Continue) sin salir del codespace.

## 2. Fine-tuning de un modelo abierto (lo que pediste)

Preparar los datos (esto sí corre en el codespace, tarda segundos):

```bash
python scripts/preparar_sft.py
# -> dataset/sft/train.jsonl y val.jsonl en formato messages
```

Validar el plan sin cargar el modelo:

```bash
python train_qlora.py --dry-run
```

Entrenar en GPU (Kaggle, ver `notebooks/kaggle_origenlm_qlora.ipynb`):

```bash
pip install -r requirements-train.txt
python train_qlora.py --epochs 1 --batch 2 --grad-acum 8 --eval-cada 50
```

Salida: `checkpoints/qlora_origen/adapter/` (un adaptador de ~40 MB) y
`meta.json` con la pérdida de validación. El adaptador se carga encima del
modelo base con `peft`; con `--fusionar` se exporta el modelo completo.

**Aviso honesto:** hoy hay 133 diálogos (~13k tokens). Eso alcanza para probar
el flujo y enseñarle estilo, no para que "sepa" conversar. Para eso hacen falta
miles de ejemplos y varias épocas.

## 3. Entrenar el modelo propio (`origen/v2.py`)

Sigue disponible y es el corazón del proyecto: arquitectura y tokenizador
propios, sin pesos externos.

```bash
python scripts/preparar_v2_es.py        # corpus + tokenizador BPE
python train_v2_bpe.py --pasos 5000     # en GPU; en CPU es lentísimo
python chat_bpe.py --checkpoint checkpoints/origen_v2_es.pt
```

Guía operativa completa en [GPU_KAGGLE.md](GPU_KAGGLE.md).

## Límites reales de hardware

| Recurso | Cuánto | Qué permite |
|---|---|---|
| Codespace 4-core | 16 GB RAM, 32 GB disco | inferencia 7B Q4, preparar datos, programar |
| Kaggle T4 | 16 GB VRAM, 30 h/semana | QLoRA de 7B; preentrenamiento del v2 |
| Codespaces Pro | ~180 core-hours/mes | 4-core consume 4 por hora → ~45 h/mes |

Crear **varias instancias no suma cómputo**: son máquinas separadas sin red
rápida entre ellas; un entrenamiento no se reparte entre codespaces. Para
escalar de verdad hace falta una GPU (Kaggle, Colab, RunPod, Lambda).
