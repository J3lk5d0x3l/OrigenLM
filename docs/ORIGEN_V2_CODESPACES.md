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
