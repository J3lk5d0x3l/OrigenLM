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

## Etapa 3 — diagnóstico de respuestas (sin más entrenamiento)

La instrucción predeterminada de `chat.py` coincide ahora con la utilizada
para crear los datos de entrenamiento. El prototipo avisa si la pregunta no
cabe completa en el contexto del checkpoint (96 bytes en las versiones de
200 y 300 pasos). Los turnos anteriores que no caben se descartan.

Tras guardar ambos checkpoints, puedes comparar respuestas con semillas
controladas, **sin modificar los pesos**:

```bash
python -m scripts.comparar_chat
```

Rutas esperadas: `/workspaces/backup-origenlm/origen_200_pasos.pt` y
`/workspaces/backup-origenlm/origen_300_pasos.pt`.
`--seed` en `chat.py` permite repetir pruebas, por ejemplo:
`python chat.py --seed 42`.

Esto no agrega respuestas predefinidas ni convierte el modelo pequeño en un
asistente útil. Para avances reales harán falta muchos más datos originales,
mejor entrenamiento orientado a conversaciones, evaluación cualitativa y un
modelo con una ventana de contexto mucho mayor.

## Etapa 4 — entrenamiento supervisado de respuestas (experimental)

La comparación de los checkpoints 200 y 300 mostró texto poco coherente. El
entrenamiento antiguo elegía fragmentos aleatorios de un flujo concatenado y
supervisaba instrucciones, preguntas y respuestas por igual, a veces sin
mostrarle al modelo la conversación completa. Para evitarlo, la etapa 4:

- Aísla cada conversación de entrenamiento.
- Calcula la pérdida **solo en bytes de respuesta del asistente y EOS**.
- Amplía el contexto a 384 bytes/tokens (RoPE no tiene posiciones aprendidas).
- Verifica que preguntas de entrenamiento y validación no se repitan.
- Crea un checkpoint *nuevo*; conserva los pesos y los respaldos existentes.

Antes de entrenar, mide la pérdida de RESPUESTAS en el checkpoint de 300 pasos:

```bash
python -m scripts.evaluar_respuestas --checkpoint /workspaces/backup-origenlm/origen_300_pasos.pt
```

Para un experimento de 100 pasos adicionales desde OrigenLM-300:

```bash
python train_respuestas.py --resume checkpoints/origen_latest.pt --steps 100
python -m scripts.evaluar_respuestas --checkpoint checkpoints/origen_respuestas.pt
python chat.py --checkpoint checkpoints/origen_respuestas.pt --seed 42
```

El entrenador escribe **solo** `checkpoints/origen_respuestas.pt`, no
`checkpoints/origen_latest.pt`. El optimizador se reinicia si se
importan pesos de la etapa anterior porque el objetivo es distinto. Si luego
quieres continuar la etapa 4, usa `--resume checkpoints/origen_respuestas.pt`
con otro destino de salida para no sobrescribir accidentalmente el archivo fuente.
El contexto del checkpoint nuevo es de 384 bytes (no palabras).

**Limitación:** 133 conversaciones siguen siendo insuficientes para una IA
general. Esta etapa busca verificar que OrigenLM aprenda el formato pregunta →
respuesta. Una pérdida baja en 15 ejemplos no demuestra comprensión general.


## OrigenLM v2 — arquitectura escalable en Codespaces (CPU)

Se ha integrado una arquitectura propia GQA con presets configurables bajo el nombre **OrigenLM v2**, sin modelos ni pesos de terceros. Esta nueva arquitectura está separada del checkpoint histórico de 300 pasos.

- `origen/v2.py`: arquitectura de OrigenLM v2.
- `train_origen_v2.py` y `chat_origen_v2.py`: entrenamiento e inferencia.
- `configs/origen_cpu.json`: configuración inicial de referencia (no representa el techo posible de la máquina).
- `docs/ENCARGO_AGENTE_CODESPACES.md`: especificación de trabajo integral para el agente de programación.
- `docs/ORIGEN_V2_CODESPACES.md`: guía de la línea v2.

**Prioridad: GitHub Codespaces CPU.** Camber no se usará salvo una futura decisión explícita. No iniciar entrenamiento grande con las 133 conversaciones de demostración.


## Comando directo y sincronización automática desde Codespaces

Instalación **una única vez** en un Codespace ya abierto:

```bash
bash scripts/instalar_origen.sh
```

En inicios posteriores del devcontainer se instala automáticamente por
`postStartCommand`. Escribe `origen` para conversar (v2 si existe
`checkpoints/origen_v2_latest.pt`, de lo contrario usa el v1), o:
`origen estado`, `origen v1`, `origen v2`, `origen entrenar --steps 1000`,
`origen pruebas`. En el chat, `/actualizar` sincroniza y recarga sin cerrar
la terminal. Para omitir la sincronización: `ORIGEN_NO_SYNC=1 origen`.

**Sincronización:** al ejecutar `origen`, el lanzador intenta
`git pull --ff-only origin main` con timeout de 20 segundos y únicamente
si está en `main` sin cambios locales en archivos rastreados. No fuerza
merges ni borra archivos, y si el repositorio diverge o está desconectado
continúa con la copia local. El proceso abierto no se reinicia de forma
automática mientras conversas: usa `/actualizar` para eso.

**GitHub Actions** ejecuta pruebas y comprobación de sintaxis en cada push
y pull request. No modifica la copia de Codespaces ni entrena pesos de forma
automática. Los checkpoints se guardan fuera de Git y no se mueven con
`git pull`.

Con una máquina de 4 vCPU y 16 GiB, el entrenador v2 detecta núcleos
efectivos y configura sus hilos CPU sin fijar el antiguo valor de 2;
el chat usa hasta 2 por defecto para minimizar latencia. No se
pretende asignar el 100 % de RAM/disco: eso provocaría falta de recursos.
El corpus disponible sigue siendo pequeño.
