# Guía de desarrollo

## Entorno

- Python 3.11 o superior.
- PyTorch (CPU en Codespaces), NumPy y `tokenizers` para el BPE propio.
- Nunca presupongas acceso a GPU, espacio o cuota ilimitada.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m origen.cli estado
```

Los cambios deben incluir pruebas que cubran el comportamiento afectado.

## Datos y tokenización

El repositorio solo contiene ejemplos didácticos. El corpus masivo de V1 y los checkpoints están excluidos de Git.

Prepara primero las fuentes cuyos derechos hayas verificado. Para utilizar las fuentes oficiales compatibles que contempla el proyecto, puedes ejecutar explícitamente:

```bash
python -m scripts.descargar_fuentes
python -m scripts.preparar_corpus --confirmar-derechos
python -m origen.cli preparar-v2
```

El último paso escribe `dataset/preparado_v2` y **no entrena pesos**. Valida licencias y procedencia antes de redistribuir datos.

## Entrenamiento y evaluación

- V1: `train_masivo.py` para corpus causal y `train_respuestas.py` para respuestas.
- V2: el preentrenamiento con BPE sigue pendiente. No mezcles el tokenizador V1 con un checkpoint V2.
- Diagnósticos: `origen benchmark`, `origen pruebas`, `origen cerrar-v1`.

No se ejecutan sesiones largas de entrenamiento en CI.

## Checkpoints y respaldos

`checkpoints/*.pt` y `dist/` no se publican en Git. Descarga cualquier respaldo ZIP generado por `origen cerrar-v1` fuera del Codespace: mantenerlo en la misma máquina no evita su pérdida si se elimina el entorno.

No subas credenciales, logs privados, datos personales, claves API ni corpus sin permiso.
