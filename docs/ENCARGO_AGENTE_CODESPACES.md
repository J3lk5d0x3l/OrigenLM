# Encargo integral al agente de GitHub Codespaces — OrigenLM

## Objetivo y restricciones irrevocables

Trabaja de forma autónoma y **edita el repositorio directamente**. Este proyecto se llama **OrigenLM**: no utilices nombres de modelos ajenos como nombre del proyecto. No descargues pesos preentrenados: entrenamiento de pesos propios, desde inicialización aleatoria. **Infraestructura principal: GitHub Codespaces, CPU. No emplees Camber ni conectes servicios de pago.** No afirmes disponer de GPU: verifica la disponibilidad real con PyTorch; Codespaces no debe depender de CUDA.

La persona quiere pasar de experimentos de 100 pasos a una canalización integral de ML que permita usar corpus completos y legales, con preparación reproducible, preentrenamiento real, evaluación, checkpoints, reanudación y chat. No llenes los discos o la cuota de horas indiscriminadamente; mide primero y aprovecha con eficiencia los núcleos disponibles.

## Estado heredado que NO debes destruir

- OrigenLM v1: `checkpoints/origen_latest.pt` (300 pasos en Codespaces, no versionado).
- Respaldos: `/workspaces/backup-origenlm/origen_200_pasos.pt` y `origen_300_pasos.pt`.
- Modelos originales en `origen/model.py` y arquitectura nueva GQA en `origen/v2.py`.
- Entrenamiento de respuestas por ejemplo en `origen/chat_training.py`.
- `dataset/chat_v2.jsonl`: solo 133 registros; **no equivale a un corpus extenso**.
- `dataset/evaluacion_v2.jsonl`: 15 preguntas reservadas.
- La CPU reportaba `GPU: False`. No existe una GPU que puedas activar desde el software.
- La raíz tiene un proyecto Python 3.11, entorno `.venv` y pruebas `python -m unittest discover -s tests -v`.

## Entrega solicitada: implementación de extremo a extremo

### Fase A: auditoría y medición de recursos (sin entrenar todavía)

1. Inspecciona Git, ramas, cambios locales, scripts, checkpoints, tests y rutas. No descartes cambios locales del usuario.
2. Ejecuta las pruebas actuales, comprueba importaciones y diagnostica recursos con `python scripts/diagnostico_cpu.py`, `nproc`, `free -h`, `df -h`, Python, PyTorch y límites cgroup.
3. Detecta núcleos efectivos disponibles y límite real de RAM. Compara 1, 2, hasta N hilos en benchmarks cortos y reproducibles; ajusta `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `torch.set_num_threads`, carga de datos y lotes. No maximices la memoria: deja margen para el sistema.
4. Estima bytes/s, tokens/s, tiempo por paso y memoria de entrenamiento de los presets `micro`, `student`, `large` y `research` si existen. No entrenes presets enormes que superen RAM o que hagan imposible terminar una etapa razonable.
5. No cambies el tipo de máquina Codespaces ni habilites gasto fuera de cuota sin autorización expresa. Si conviene pasar a 4 u 8 núcleos, informa disponibilidad y efecto sobre horas-núcleo sin efectuar cambios.

### Fase B: biblioteca de datos propia y reproducible

1. Diseña una canalización para corpus **completos** de texto autorizado: `.txt`, `.jsonl`, `.jsonl.gz`, `.zip` o fuentes públicas explícitamente permitidas con licencia identificable. No distribuyas ni descargues masivamente libros protegidos sin autorización.
2. Usa un **manifiesto de fuentes** con URL, licencia, fecha, procedencia, idiomas, tamaño, SHA256 y atribución cuando corresponda. Por defecto trabaja con los archivos locales provistos; ofrece comandos explícitos para descargar fuentes públicas aprobadas, con comprobación de cuota/disco.
3. Procesa con streaming/chunks para no cargar toda una biblioteca en RAM. Normalización UTF-8, segmentación, deduplicación exacta y aproximada razonable, filtrado de vacíos y basura, preservación de documentos y metadatos.
4. Divisiones entrenamiento/validación/prueba **por documento o fuente**, antes de trocear en tokens, para evitar contaminación. Detecta solapamientos; no metas preguntas de evaluación en entrenamiento.
5. Separa corpus de **preentrenamiento** (texto continuo) y datos de **instrucción** (pregunta/respuesta). Documenta tamaño real en bytes, documentos y tokens. No declares tener millones de tokens si no los mediste.

### Fase C: tokenizador + arquitectura OrigenLM v2

1. Revisa la arquitectura existente `origen/v2.py`: GQA, RoPE, SDPA, SwiGLU y causalidad. Corrige bugs demostrables, con tests.
2. El tokenizador v1 por byte de vocabulario 262 es muy ineficiente para una biblioteca en español. Añade **tokenizador subword entrenado desde cero sobre el corpus autorizado**, por ejemplo BPE, sin vocabularios ni pesos de terceros. Documenta cómo guardar `tokenizer.json` y cómo emparejarlo con checkpoints.
3. Haz que v2 admita el vocabulario nuevo de forma explícita y verificable; evita que un checkpoint sea cargado con tokenizador incompatible. Mantén intactos el v1 y sus pesos; no intentes cargar directamente un checkpoint con distinta dimensión de vocabulario.
4. Mantén presets y configuración por JSON; selecciona dimensiones según el benchmark medido. Un modelo grande sin datos suficientes no se considera avance.
5. Garantiza que el contexto, sus límites, el enmascaramiento de la pérdida del asistente y la terminación EOS sean correctos. No presentes memoria o longitud infinitas.

### Fase D: preentrenamiento continuo optimizado para CPU

1. Implementa entrada de tokens a disco (mmap o shards), lectura con batches y sampling que no mezcle documentos incorrectamente, evitando cargar el corpus íntegro en RAM.
2. Entrenamiento causal autoregresivo **de texto completo** para aprender patrones generales; posteriormente ajuste conversacional con loss solo en la respuesta. Ambos objetivos han de estar separados y probados.
3. Optimiza para CPU real: hilos adecuados, dataloader/prefetch donde mejore, grad accumulation, secuencias, checkpointing, optimizador y mixed precision solo si funciona en este CPU y se demuestra estable. Compara `torch.compile` vs eager antes de adoptarlo; evita compilaciones que consuman la cuota.
4. Registra cada N pasos: loss entrenamiento y validación, tokens/s reales, tiempo/step, uso de memoria y parámetros contados. Guarda checkpoints **atómicos** y reanudables incluyendo modelo, optimizador, scheduler, step, tokenizer/config, RNG y estado de datos cuando sea factible.
5. Añade control de pérdidas NaN, fallos de RAM/disk y corte por tiempo/cuota, con capacidad de reanudar. No sobrescribas `checkpoints/origen_latest.pt`.
6. Prepara un comando integral como `bash scripts/entrenamiento_completo_codespaces.sh --dry-run` y su modo `--run`, con diagnósticos, preparación, benchmark, train, evaluación y reporte. No hagas trabajos masivos por defecto al importar un módulo.
7. Después de benchmarks y tests, **ejecuta un entrenamiento auténtico y acotado** en la máquina actual con los datos autorizados disponibles, solo si la cuota y el disco lo permiten. Deja el checkpoint nuevo y métricas reproducibles. No confundir entrenamiento real con pruebas sobre 133 registros.

### Fase E: chat, evaluación y entregables

1. Integra una interfaz de chat con el checkpoint v2 y su tokenizador propio; historial limitado explícitamente por contexto, manejo de errores Unicode, argumentos configurables.
2. Añade evaluaciones independientes de lenguaje, coherencia, saludo, identidad, español, instrucciones y programación básica. No uses exclusivamente la pérdida como medida de calidad.
3. Ejecuta `python -m unittest discover -s tests -v` y una prueba end-to-end pequeña. No anuncies éxito de funciones no ejecutadas.
4. Escribe un `docs/INFORME_ENTRENAMIENTO.md` con hardware real, datasets usados, licencias, métricas, parámetros, memoria, tokens/s, commits, problemas y decisiones.
5. Documenta el comando **exacto** para reanudar el entrenamiento y el **comando exacto** para abrir el chat. Muestra un ejemplo real de respuesta generado, sin inventarlo.
6. Trabaja en commits pequeños, haz push al repositorio, deja `main` funcional y no entregues únicamente archivos sueltos. Si no tienes permiso para hacer push, informa commit/hash y el comando de publicación.
7. Devuelve una respuesta compacta al terminar, con: estado de cada fase, commit(s), hardware detectado, total de datos reales, tokens/s, pruebas, archivos y ruta del checkpoint, resultado del chat y siguiente comando. Si alguna fase no se logra, indícalo y la causa técnica.

## Prohibiciones y criterios de éxito

No usar Camber, no configurar servicios externos de pago, no copiar pesos de Claude u otros LLM, no usar nombres de modelos ajenos en el branding, no borrar respaldos del usuario, no subir secretos, no prometer capacidad de frontera ni GPU en Codespaces. No iniciar descargas enormes de internet sin verificar licencia, espacio y cuota.

**Resultado esperado:** OrigenLM con infraestructura CPU de principio a fin, lista para corpus extensos legales y entrenamiento largo reproducible, evaluado con datos separados y conservando checkpoints históricos. La calidad no está garantizada hasta verla en pruebas reales.
