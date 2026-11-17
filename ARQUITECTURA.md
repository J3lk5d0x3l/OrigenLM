# Arquitectura de OrigenLM

Modelo Transformer autoregresivo (decoder-only) entrenado desde pesos aleatorios.
No utiliza pesos de Claude, GPT, Llama, Qwen ni modelos comerciales.

## Componentes
- tokenizer propio UTF-8 con 256 tokens byte y seis tokens de control.
- embeddings entrenables, RMSNorm, atención causal con RoPE y SwiGLU.
- entrenamiento con AdamW, clipping de gradientes, checkpoints recuperables.
- interfaz conversacional por terminal, pruebas automatizadas.

## Limitaciones actuales
- Contexto de apenas 96 tokens byte en el preset inicial.
- Datos demostrativos escasos, incapaces de generar una IA conversacional útil.
- Sin evaluación formal, alineación, herramientas, memoria externa ni KV-cache.
- El entrenamiento incluido es una prueba funcional, no un LLM de nivel Claude.

## Próximas fases
1. Recopilar datos propios y con licencias compatibles.
2. Implementar BPE y contexto largo.
3. Entrenamiento medido con evaluación separada y GPUs si procede.
4. Supervisión conversacional, pruebas de seguridad y razonamiento.
5. API y aplicación local, herramientas con permisos y almacenamiento de checkpoints.

El checkpoint de ejemplo del ZIP no se sube a GitHub porque .gitignore
excluye archivos .pt; los nuevos pesos se generan en Codespaces al entrenar.
