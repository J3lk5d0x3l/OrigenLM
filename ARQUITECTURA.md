# Arquitectura de OrigenLM

OrigenLM es un proyecto experimental de entrenamiento desde cero. La implementación usa técnicas publicadas de redes Transformer y no incorpora pesos externos.

## V1: referencia entrenada

- Arquitectura decoder-only con atención causal, RoPE, RMSNorm y SwiGLU.
- Tokenizador UTF-8 por bytes: 256 valores y seis tokens de control.
- Configuración de la primera generación: 357.600 parámetros.
- Preentrenamiento causal con corpus separado por documentos; ajuste de respuestas posterior.
- El ajuste final fue registrado en el paso 20.800, pero la evaluación de conversación sigue mostrando errores importantes.
- Los checkpoints son locales y están ignorados por Git. El repositorio NO contiene los pesos.

## V2: trabajo en curso

- `origen/v2.py`: arquitectura propia con atención agrupada (GQA), RoPE, SwiGLU y KV-cache para inferencia.
- `origen/tokenizer_v2.py`: tokenizador BPE que aprende un vocabulario desde datos autorizados.
- `scripts/preparar_v2.py`: crea un corpus V2 propio, sin reutilizar vocabularios externos, manteniendo los splits train/val.
- **Pendiente:** conectar de forma segura el vocabulario BPE con el preentrenador V2, generar checkpoints con identidad de tokenizador y ejecutar evaluación independiente.
- `train_origen_v2.py` aún usa la vía experimental de entrenamiento conversacional con el tokenizador por bytes. No debe presentarse como entrenamiento BPE completo.

Cambiar el vocabulario o el tamaño de arquitectura impide cargar directamente los pesos de V1 en V2.

## Datos, experimentos y seguridad

1. Usar exclusivamente datos propios o reutilizables con permiso, registrando fuente y licencia.
2. Separar datos de entrenamiento, validación y prueba **antes** de entrenar el tokenizador.
3. Guardar checkpoints fuera del control de versiones; respaldar los pesos fuera de Codespaces.
4. Evaluar precisión y coherencia, además de la NLL. Una menor NLL no demuestra razonamiento.
5. Mantener herramientas externas tras permisos explícitos y controles de seguridad.

Los detalles de instalación y pruebas están en [docs/DESARROLLO.md](docs/DESARROLLO.md).
