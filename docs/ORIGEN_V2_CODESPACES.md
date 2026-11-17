# OrigenLM v2 — arquitectura escalable sobre CPU Codespaces

Esta variante usa Transformer GQA, RoPE, SwiGLU, SDPA y pesos aleatorios. No es un modelo Claude. No utiliza pesos externos ni derivados.

## Uso actual: SOLO GitHub Codespaces

**No usar Camber ni GPU como parte de este plan.** Codespaces tiene CPU. El tamaño de la VM puede ampliarse cuando la cuenta lo permita, a costa de gastar más horas-núcleo por hora. Revisa cuota y presupuesto antes de cambiarla.

Pruebas:

```bash
python -m unittest discover -s tests -v
python scripts/diagnostico_cpu.py
```

Entrenamiento de arquitectura v2, independiente del checkpoint histórico de 300 pasos:

```bash
python train_origen_v2.py --config configs/origen_cpu.json --steps 1000 --batch-size 2 --grad-accum 4 --save-every 100 --output checkpoints/origen_v2_latest.pt
python chat_origen_v2.py --checkpoint checkpoints/origen_v2_latest.pt
```

**No ejecutar aún entrenamiento largo:** `dataset/chat_v2.jsonl` solo contiene 133 ejemplos, no es un corpus de biblioteca. Antes hay que implementar una ingesta de datos legítimos, limpieza, tokenizador eficiente, división de entrenamiento/validación, benchmark CPU y seguimiento de métricas. El encargo integral para el agente está en `docs/ENCARGO_AGENTE_CODESPACES.md`.

La arquitectura v2 no puede cargar directamente los pesos v1; mantiene `checkpoints/origen_latest.pt` y respaldos intactos.

## Inferencia con KV-cache (etapa 5)

`OrigenV2.generate` reutiliza las claves y valores ya calculados en lugar de
reprocesar el prompt en cada token. El coste pasa de cuadrático a lineal sobre
la longitud generada, que es lo que hacía inviable el chat en CPU.

- `model.new_cache()` crea un `KVCache` por capa; se guarda tras RoPE y antes
  de repetir los grupos de GQA, así que ocupa `n_kv_heads`, no `n_heads`.
- Cuando el cache llena `max_seq_len` se descarta la posición más antigua.
- `generate(..., use_cache=False)` conserva la ruta anterior para comparar.
- El KV-cache es solo de inferencia: pasar `targets` junto a `caches` falla.
- `tests/test_kv_cache.py` verifica que los logits incrementales coinciden con
  el forward completo y que la generación greedy es idéntica en ambas rutas.

Las tablas de RoPE se calculan una vez por `(head_dim, dispositivo, dtype)` en
`rope_tables()`, en lugar de recalcularse en cada capa de cada forward.

## Programación del learning rate

`train_origen_v2.py` acepta `--warmup` (por defecto 50 pasos) y
`--min-lr-ratio` (0.1): calentamiento lineal y después coseno. El weight decay
se aplica solo a las matrices; RMSNorm y embeddings quedan excluidos porque
`lm_head` está atado a `tokens` y se penalizaría dos veces.

```bash
python train_origen_v2.py --config configs/origen_cpu.json --steps 1000 \
  --warmup 50 --batch-size 2 --grad-accum 4 --save-every 100 \
  --output checkpoints/origen_v2_latest.pt
```

Estas mejoras no aumentan los datos disponibles: `dataset/chat_v2.jsonl` sigue
teniendo 133 ejemplos y 7.466 bytes supervisados. Siguen pendientes las fases
B y D del encargo (corpus propio, BPE y preentrenamiento continuo).
