# OrigenLM-Mythos — modelo propio, desde cero

**Estado:** implementación inicial en el repositorio; pesos sin entrenar. No es
Claude Mythos ni reproduce su arquitectura privada, entrenamiento, capacidades
o pesos. «Mythos» es solo un nombre para esta línea experimental de OrigenLM.

## Arquitectura

- Transformer autoregresivo decoder-only, escrito en PyTorch.
- RMSNorm, RoPE, atención causal eficiente (SDPA) y Grouped Query Attention.
- SwiGLU, embeddings enlazados y tokenizador original por bytes UTF-8.
- Inicialización de todos los pesos desde números aleatorios: no se descargan pesos.
- Presets escalables: micro, student, l4, research, o JSON propio.
- La longitud de generación se configura con --max-new-tokens; el contexto
  también se configura, pero ningún hardware ofrece memoria o contexto infinito.
- La generación larga usa una ventana deslizante y olvida tokens anteriores:
  no equivale a razonamiento infinito ni a memoria ilimitada.

## Primer experimento en Codespaces (CPU)

En la raíz de OrigenLM:

```bash
python -m unittest discover -s tests -v
python train_mythos.py --preset micro --steps 10 --save-every 10 --output checkpoints/mythos_micro.pt
python chat_mythos.py --checkpoint checkpoints/mythos_micro.pt --seed 42
```

El ejemplo solo demuestra que el sistema funciona. Una respuesta sensata a
«Hola» requiere más datos y entrenamiento; con 133 ejemplos no esperes calidad
de un asistente moderno.

## Ejecución en Camber Cloud (GPU L4, cuando esté conectado)

Selecciona **GPU · Xsmall / L4** y configura un entorno con PyTorch compatible
con la GPU. Comprueba antes de entrenar:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sin GPU')"
```

Después de tener el repositorio y sus datos disponibles en Camber:

```bash
python train_mythos.py --config configs/mythos_l4.json --steps 100 --batch-size 2 --grad-accum 2 --save-every 25 --output checkpoints/mythos_l4.pt
```

**No ejecutes el entrenamiento si GPU=False:** así evitarás consumir las
horas de Camber sin aceleración. El beneficio Student tiene una cuota de
recursos; consulta el panel de uso antes de ejecutar sesiones largas.

La salida se guarda **por separado** de OrigenLM-300. Si deseas continuar:

```bash
python train_mythos.py --resume checkpoints/mythos_l4.pt --output checkpoints/mythos_l4.pt --steps 100
```

No se modifica `checkpoints/origen_latest.pt` ni los respaldos de
200/300 pasos. Los checkpoints OrigenLM anteriores **no son compatibles
directamente** con esta arquitectura; Mythos empieza con pesos nuevos.

## Escalar sin topes artificiales

Copia `configs/mythos_l4.json` y modifica las dimensiones, capas,
cabezas y el contexto según memoria real. Todos los valores pasan por
validación; un valor mayor puede producir error por falta de RAM o VRAM.
Un contexto grande consume más memoria y tiempo; una GPU L4 no puede
entrenar un modelo de escala frontera por sí sola.

El código supervisa las respuestas del asistente, usando el mismo esquema
que `origen/chat_training.py`, y reserva ejemplos de validación.
Para conseguir calidad real se necesita mucho más que una arquitectura:
corpus amplio y licenciado, entrenamiento largo, métricas, evaluación
de razonamiento y programación, y sistemas externos de recuperación/herramientas.

No se han integrado agentes, navegador, herramientas externas ni ejecución
de código por parte del modelo; son proyectos posteriores y distintos del
entrenamiento del modelo base.
