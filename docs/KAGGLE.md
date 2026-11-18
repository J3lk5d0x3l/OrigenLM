# Entrenar OrigenLM en Kaggle y guardar el adaptador en Hugging Face

Flujo: **Codespace** (editar y validar datos) → **Kaggle** (entrenar con GPU) →
**Hugging Face Hub** (guardar el adaptador).

El modelo base (Qwen2.5-Coder-7B-Instruct) no se guarda en ningún lado: se descarga
solo desde Hugging Face. Lo único que entrenas y guardas es el adaptador LoRA
(unas decenas de MB).

## Una sola vez

1. Crea una cuenta en <https://huggingface.co> y un token con permiso **Write**
   (Settings → Access Tokens).
2. En Kaggle importa `notebooks/kaggle_origenlm_qlora.ipynb` (Create → Notebook →
   File → Import Notebook).
3. En el notebook: **Settings → Accelerator: GPU T4 x2** e **Internet: ON**.
4. **Add-ons → Secrets**: agrega el secreto `HF_TOKEN` con tu token y actívalo para
   este notebook.

## Cada vez que quieras entrenar

1. En el Codespace agrega datos (ver `dataset/entradas/LEEME.md`) y valida:

   ```bash
   python scripts/preparar_sft.py
   python train_qlora.py --dry-run
   git add -A && git commit -m "más datos" && git push
   ```

2. En Kaggle corre el notebook completo. Baja el repo desde `main`, entrena y deja el
   adaptador en `/kaggle/working/qlora_origen`.
3. La penúltima celda lo sube a `https://huggingface.co/<tu-usuario>/origenlm-qlora`
   (privado por defecto).

## Usar el adaptador

```bash
# bajarlo (en el codespace o en tu PC)
export HF_TOKEN=hf_...
python scripts/subir_hf.py --bajar --repo origenlm-qlora --carpeta checkpoints/qlora_origen
```

```python
# cargarlo sobre el modelo base (necesita GPU para el 7B)
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
modelo = PeftModel.from_pretrained(base, "checkpoints/qlora_origen/adapter")
```

Para correrlo en CPU (codespace) fusiona y convierte a GGUF; los pasos están al final
del notebook.

## Qué esperar

Con pocos datos el modelo aprende **estilo**, no conocimiento nuevo. El script avisa
si hay menos de ~20k tokens. Lo que más mejora el resultado es sumar ejemplos buenos
de pregunta y respuesta sobre tus propios proyectos.
