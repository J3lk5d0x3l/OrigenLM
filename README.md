# OrigenLM — Inteligencia Artificial Propia

Proyecto experimental de un modelo de lenguaje tipo Transformer entrenado **desde cero**.
No utiliza pesos preentrenados de Claude, Llama, Qwen u otros modelos.
La arquitectura utiliza técnicas públicas conocidas, implementadas en este repositorio.

> **Estado:** prototipo didáctico pequeño. El dataset incluido NO es suficiente para
> generar un asistente útil: las respuestas pueden ser incoherentes.

## GitHub Codespaces (recomendado para el beneficio GitHub Student)

Este repositorio incluye `.devcontainer/devcontainer.json` que configura Python
3.11 y, al abrir un Codespace, ejecuta `scripts/bootstrap_codespaces.sh`.
Instala PyTorch CPU y ejecuta pruebas, **sin iniciar un entrenamiento automáticamente**.

1. Abre el repositorio → **Code** → **Codespaces** → **Create codespace on main**.
2. Cuando termine la instalación y pruebas, ejecuta:

```bash
bash scripts/entrenar_codespaces.sh 100
python chat.py
```

El primer entrenamiento genera `checkpoints/origen_latest.pt`.
Si existe un checkpoint, el script continúa su entrenamiento automáticamente.

**Para evitar cargos:** usa 2 núcleos, detén Codespaces al terminar, revisa tu
cuota de horas-núcleo y configura un presupuesto con límite de gasto.
Codespaces proporciona CPU remota, no GPU garantizada.

## Camber Cloud (integración preparada)

La documentación de conexión segura por OAuth, empaquetado del proyecto y diagnóstico
está en [docs/CAMBER.md](docs/CAMBER.md). La autorización y el despliegue requieren
que el propietario de la cuenta complete el inicio de sesión en Camber.

```bash
python3 scripts/diagnostico_entorno.py
python3 scripts/preparar_camber.py
```

## Ejecutar en tu PC

Instala Python 3.10+ y PyTorch CPU desde el índice oficial:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
python train.py --preset tiny --steps 100 --batch-size 2 --seq-len 96
python chat.py
```

## Datos de entrenamiento

Los ejemplos originales de prueba están en `dataset/chat.jsonl`; cada línea:

```json
{"user":"Hola","assistant":"¡Hola! Soy Origen."}
```

El modelo también acepta un corpus original en `dataset/corpus.txt`:

```bash
python train.py --data dataset/corpus.txt --steps 300
```

Entrena exclusivamente con datos autorizados; nunca pongas tokens ni datos sensibles.

## Presets y arquitectura

- **tiny:** ~0.4 M parámetros, pruebas CPU.
- **small:** ~1.8 M, necesita más recursos.
- **medium:** ~11 M, mayor coste de entrenamiento.

Código: `origen/model.py`, `origen/tokenizer.py`, `train.py`, `chat.py`.
Consulta `ARQUITECTURA.md` para la hoja de ruta.

## Checkpoints

Los archivos `checkpoints/*.pt` están excluidos deliberadamente de Git.
No se suben los pesos experimentales de 120 pasos del ZIP, pero puedes
entrenar un modelo completamente nuevo dentro de Codespaces.
Respalda cualquier checkpoint valioso antes de eliminar su Codespace.

## Verificación

```bash
python -m unittest discover -s tests -v
```

## Etapa 2 — dataset ampliado y evaluación separada

Se añadió un dataset **didáctico** con 133 ejemplos originales
creados para OrigenLM y 15 ejemplos reservados de evaluación.
No es suficiente para un asistente general: el checkpoint de 200 pasos sigue
siendo experimental y el contexto del modelo tiny actual es de solo 96 bytes.

Primero mide el checkpoint existente (no altera sus pesos):

```bash
python -m scripts.evaluar_checkpoint
```

Para un **experimento posterior**, realiza una copia del checkpoint y entrena
con el dataset ampliado. El comando original por defecto sigue usando
`dataset/chat.jsonl`; no hay entrenamientos automáticos.

```bash
mkdir -p /workspaces/backup-origenlm
cp checkpoints/origen_latest.pt /workspaces/backup-origenlm/origen_200_pasos.pt
python train.py --resume checkpoints/origen_latest.pt --data dataset/chat_v2.jsonl --steps 100 --batch-size 2 --seq-len 96
python -m scripts.evaluar_checkpoint
```

No compares directamente la pérdida de entrenamiento con la de evaluación:
se calculan en muestras distintas, con distribuciones y condiciones de
contexto que pueden diferir. Para medir una mejora, compara el mismo
conjunto de evaluación antes y después de entrenar.
