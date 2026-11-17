# Entrenar OrigenLM V2 en GPU gratuita (Kaggle T4)

Guía operativa. **Nada de esto se ejecuta solo y nada se sube sin tu permiso.**

## 1. Qué se transfiere y por qué

| Archivo | Tamaño | Contenido |
|---|---|---|
| `train.bin` | 213,6 MiB | 111.967.457 tokens uint16 (Wikipedia es, CC BY-SA 4.0) |
| `val.bin` | 9,2 MiB | 4.829.751 tokens reservados |
| `tokenizer.json` | 0,5 MiB | BPE propio de 8.192 tokens |
| `meta.json` | ~0,3 MiB | recuentos, hashes y particiones |
| `origen_v2_es.pt` | 78 MiB | pesos + estado de AdamW + paso |

Son datos **tuyos o derivados de Wikipedia**. Kaggle los guarda privados por
defecto, pero siguen siendo una subida a un servicio externo.

## 2. Empaquetar en el Codespace (no sube nada)

```bash
cd /workspaces/OrigenLM
python -m scripts.exportar_checkpoint --salida dist --usuario TU_USUARIO_KAGGLE
```

Deja `dist/corpus/`, `dist/checkpoint/`, `dist/hashes.json` y
`dist/MANIFIESTO.md`. Verifica SHA-256 y se niega a exportar si el checkpoint no
corresponde a ese tokenizador.

## 3. Subir (solo si lo autorizas)

```bash
pip install kaggle            # y tu kaggle.json en ~/.kaggle/
kaggle datasets create -p dist/corpus --dir-mode zip
kaggle datasets create -p dist/checkpoint --dir-mode zip
```

## 4. Ejecutar el notebook

1. Kaggle → *Create* → *Notebook* → **File > Import Notebook** →
   `notebooks/kaggle_origenlm_v2_t4.ipynb`.
2. *Settings*: Accelerator **GPU T4 x2**, Internet **ON**.
3. Añade los dos datasets privados con *Add Input*.
4. Ajusta `COMMIT` (SHA de `main`) y los dos slugs `/kaggle/input/...`.
5. Ejecuta las celdas: verifica hashes → benchmark de 30 pasos → entrenamiento
   con `--max-minutos 55` → evaluación.

## 5. Qué hace el entrenador en GPU

- `--device cuda --precision auto`: bf16 si la tarjeta lo soporta; en T4
  (compute 7.5) no hay bf16, así que cae a fp16 con `GradScaler`.
- Guarda cada 100 pasos de forma atómica (`.tmp` + `os.replace`).
- `SIGTERM`/`SIGINT` guardan el paso en curso antes de salir: cubre el stop de
  Kaggle y el límite de inactividad.
- `--resume` restaura pesos, AdamW y el punto exacto de la curva de LR.
- La evaluación se hace siempre en fp32 para que el NLL sea comparable con CPU.

## 6. Límites reales de Kaggle

- 30 h de GPU por semana (T4 x2 o P100).
- Sesión máxima ~9 h; el guardado por señal evita perder trabajo.
- El notebook escribe solo en `/kaggle/working`; hay que descargar el
  checkpoint desde *Output* antes de cerrar.

## 7. Reanudar en otra sesión

Misma celda de entrenamiento: `--resume` continúa desde el paso guardado. El
calendario del coseno se conserva porque el total vive en el checkpoint
(`total_steps`). Si quieres alargar el entrenamiento, usa
`--pasos-total 12000` (o el horizonte que decidas): la curva se estira, pero el
warmup **no** se reinicia.
