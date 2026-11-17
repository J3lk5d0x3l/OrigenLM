<div align="center">

# OrigenLM

**Un modelo de lenguaje abierto, creado en Colombia.**

Investigación reproducible · Pesos propios · Comunidad latinoamericana

[![Pruebas](https://github.com/J3lk5d0x3l/OrigenLM/actions/workflows/tests.yml/badge.svg)](https://github.com/J3lk5d0x3l/OrigenLM/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Licencia](https://img.shields.io/badge/licencia-MIT-555555)](LICENSE)

[**Comenzar**](#comenzar) · [**Arquitectura**](ARQUITECTURA.md) · [**Colaborar**](CONTRIBUTING.md) · [**Discord**](https://discord.biron.net)

</div>

---

### ¿Qué es OrigenLM?

OrigenLM es un proyecto de investigación para construir y entrenar un modelo de lenguaje **con pesos propios**, sin depender de checkpoints preentrenados de terceros.

El objetivo es desarrollar tecnología de IA accesible, verificable y útil para personas y desarrolladores de Colombia y Latinoamérica.

> **Estado: experimental.** El primer modelo (V1, 357.600 parámetros) completó su preentrenamiento y ajuste conversacional, pero todavía produce respuestas incorrectas o incoherentes. No es un sustituto de asistentes comerciales.

### Estado del proyecto

| Componente | Estado |
|---|---|
| OrigenLM V1 · Transformer y entrenamiento desde cero | Implementado y probado |
| Checkpoints, corpus y evaluación separada | Implementados |
| Herramientas locales, web y conectividad MCP | Prototipos experimentales |
| OrigenLM V2 · arquitectura GQA y KV-cache | Prototipo disponible |
| Tokenizador BPE propio y preparación de corpus V2 | Disponibles |
| Preentrenamiento completo de V2 con BPE | Pendiente |
| Aplicaciones web, móvil y escritorio | Proyectadas |

**Nota:** V2 aún no está entrenado con el tokenizador BPE; tener su código no significa disponer de un nuevo modelo funcional. Los pesos V1 se conservan por separado.

### Comenzar

Necesitas Python **3.11 o superior**. Clona el repositorio y ejecuta:

```bash
git clone https://github.com/J3lk5d0x3l/OrigenLM.git
cd OrigenLM
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m origen.cli estado
```

En GitHub Codespaces, el lanzador `origen` se instala al iniciar el contenedor.

| Comando | Función |
|---|---|
| `origen estado` | Ver recursos y checkpoints |
| `origen benchmark` | Comparar rendimiento en CPU |
| `origen cerrar-v1` | Evaluar V1 y crear respaldo local |
| `origen preparar-v2` | Crear vocabulario BPE y shards V2 desde fuentes autorizadas |
| `origen pruebas` | Ejecutar pruebas |

Para entrenar un **modelo de demostración desde cero** (no el V1 entrenado del equipo):

```bash
python train.py --preset tiny --steps 100 --batch-size 2 --seq-len 96
python chat.py --checkpoint checkpoints/origen_latest.pt
```

Los datasets externos, los pesos entrenados y los respaldos **no están incluidos** en el repositorio. La preparación de V2 necesita un corpus local cuyo uso esté autorizado. Consulta [desarrollo](docs/DESARROLLO.md) para detalles.

### Proyecto

```text
origen/     Modelos, tokenizadores, agentes y utilidades
scripts/    Preparación de datos, benchmarks y diagnósticos
train*.py   Entrenamiento experimental
chat*.py    Inferencia local por terminal
dataset/    Ejemplos pequeños de demostración y evaluación
tests/      Pruebas automáticas
docs/       Guía de desarrollo
```

### Colaborar

Las contribuciones de código, pruebas, documentación y datasets **con derechos de uso verificables** son bienvenidas. Antes de abrir un pull request, lee [CONTRIBUTING.md](CONTRIBUTING.md).

Para reportar una vulnerabilidad consulta [SECURITY.md](SECURITY.md); no publiques credenciales ni detalles sensibles en issues.

### Comunidad

¿Quieres aportar a una IA desarrollada desde Colombia?

**[Únete a Discord — discord.biron.net](https://discord.vyrron.net)**

<sub>Código bajo licencia [MIT](LICENSE). Los derechos y condiciones de uso de fuentes externas y checkpoints son independientes.</sub>
