# OrigenLM

Asistente de código y conversación en español, construido sobre modelos abiertos
**Qwen2.5-Coder** (licencia Apache 2.0) con un flujo propio de datos, evaluación y
fine-tuning.

El proyecto dejó atrás la etapa de entrenar una arquitectura desde cero: ahora se
apoya en pesos abiertos ya entrenados y concentra el trabajo en los datos, el
ajuste fino y el uso local.

## Plataforma nueva: chat y backend listos para Codespaces

OrigenLM integra **AnythingLLM** (interfaz y API) con **Ollama** (modelo local ligero). El entrenamiento QLoRA en Kaggle sigue siendo independiente.

En el Codespace ejecuta **Terminal → Run Task → All Services (OrigenLM)**, o desde la terminal:

```bash
bash scripts/start_services.sh
```

Luego abre el **puerto privado 3001** en **Ports** y completa la creación del administrador. El modelo inicial es `qwen2.5-coder:1.5b` para ajustarse a 4 CPU / 16 GB RAM. En el panel cambia la marca a **OrigenLM** desde Settings → Customization → Branding & Whitelabeling.

**Guía completa:** [docs/PLATAFORMA.md](docs/PLATAFORMA.md). Esta integración utiliza la imagen upstream de AnythingLLM; no reemplaza ni borra los scripts existentes. El adaptador de Kaggle todavía requiere exportación a un formato compatible antes de poder usarlo aquí.

## Qué hace hoy

| Camino | Comando | Dónde corre |
|---|---|---|
| Chatear | `bash scripts/chat_local.sh` | codespace / CPU (≈6 tok/s) |
| Preparar datos | `python scripts/preparar_sft.py` | cualquier lado |
| Fine-tuning (QLoRA) | `python train_qlora.py` | GPU (Kaggle T4) |

La primera vez, el chat descarga el modelo (~4,4 GB) en `models/`, que no se
versiona. Guía completa: [ENTRENAMIENTO.md](docs/ENTRENAMIENTO.md).

## Requisitos

- Python 3.11 o superior.
- Para chatear: nada más (llama.cpp se descarga solo).
- Para entrenar: GPU con 16 GB de VRAM o más (Kaggle T4 alcanza).

```bash
git clone https://github.com/J3lk5d0x3l/OrigenLM.git
cd OrigenLM
```

## Estructura

```text
chat_local.py        Chat con el modelo abierto (llama.cpp)
train_qlora.py       Fine-tuning QLoRA sobre Qwen2.5-Coder
scripts/
  chat_local.sh      Arranca el servidor local y abre el chat
  preparar_sft.py    Convierte diálogos al formato de entrenamiento
dataset/             Diálogos de ejemplo
docs/                Guía de entrenamiento
notebooks/           Notebook listo para Kaggle (GPU T4)
models/ tools/       Pesos y binarios descargados (ignorados por Git)
```

## Uso desde OpenCode

`scripts/chat_local.sh` levanta una API compatible con OpenAI en
`http://127.0.0.1:8080/v1`. Cualquier cliente compatible (OpenCode, Cline,
Continue) puede usarla como proveedor.

## Licencia

Código bajo [MIT](LICENSE). Los pesos de Qwen2.5-Coder pertenecen a Alibaba
Cloud y se distribuyen bajo Apache 2.0.
