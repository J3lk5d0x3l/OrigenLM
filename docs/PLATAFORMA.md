# Plataforma OrigenLM: Codespaces + Kaggle

## Arquitectura de desarrollo
- **AnythingLLM** (imagen oficial `mintplexlabs/anythingllm`): frontend web, backend, cuentas, chat, documentos y RAG.
- **Ollama** (imagen oficial `ollama/ollama`): inferencia en CPU dentro de Codespaces.
- **Kaggle**: entrenamiento QLoRA con GPU T4 usando `notebooks/kaggle_origenlm_qlora.ipynb`. No se ejecuta dentro de Docker Compose.

El código completo de AnythingLLM **no está copiado dentro de este repositorio**: se usa su contenedor oficial. Su UI se puede personalizar desde el panel; para modificaciones profundas al código habría que desarrollar un fork propio. Proyecto original: https://github.com/Mintplex-Labs/anything-llm (MIT).

## Iniciar con un clic en Codespaces
1. Abre `J3lk5d0x3l/OrigenLM` en GitHub Codespaces. Si el Codespace ya está abierto, ejecuta `git pull origin main`.
2. En VS Code, selecciona **Terminal → Run Task → All Services (OrigenLM)**.
3. Docker Compose descarga las imágenes e inicia AnythingLLM y Ollama. La primera ejecución descarga **qwen2.5-coder:1.5b**. Deja que la tarea termine; necesita conexión y espacio de disco.
4. Ve a **Ports**, abre el puerto **3001** y déjalo en visibilidad **Private**. Completa el asistente inicial, crea una cuenta administradora segura y habilita autenticación.
5. En **Settings → AI Providers → LLM**, confirma **Ollama**, la URL `http://ollama:11434` y el modelo `qwen2.5-coder:1.5b` si no aparecen por defecto.
6. Para el nombre/logotipo, entra como administrador a **Settings → Customization → Branding & Whitelabeling**. Cambia la identidad, título de pestaña, paleta y favicon a **OrigenLM**.

También puedes usar directamente:
```bash
bash scripts/start_services.sh
bash scripts/status_services.sh
bash scripts/stop_services.sh
```

## Recursos
Codespaces propuesto: **4 CPU / 16 GB RAM, sin GPU**. Por eso se usa un modelo inicial ligero (1.5B) para probar que frontend/backend responden. Los modelos de 7B en CPU pueden ser notablemente más lentos y consumir más memoria. Antes de cambiar a un modelo pesado revisa disco y RAM. Las imágenes de Docker, modelos y vectores ocupan almacenamiento adicional.

Para escoger otro modelo **antes de iniciar por primera vez**:
```bash
ORIGEN_MODEL=qwen2.5-coder:3b bash scripts/start_services.sh
```
Si cambias de modelo después de configurar la aplicación, selecciónalo también en la interfaz de AnythingLLM. No se publica el puerto de Ollama; solo está disponible en la red interna de Docker.

## Cómo encaja el entrenamiento Kaggle
El flujo QLoRA existente sigue intacto en `train_qlora.py` y `notebooks/kaggle_origenlm_qlora.ipynb`. El adaptador PEFT producido en `/kaggle/working/qlora_origen/adapter` **no es un modelo directamente cargable por Ollama**. Para usarlo como el modelo de chat de OrigenLM se deberá fusionar con la base compatible, convertir a GGUF, cuantizar, guardar el artefacto en almacenamiento persistente (p. ej., Hugging Face o una versión de Kaggle) y registrarlo en Ollama. Esta integración/exportación aún no está automatizada. No confundas el modelo ligero de arranque con el adaptador entrenado en Kaggle.

## Persistencia, seguridad y mantenimiento
- Datos de AnythingLLM: `.runtime/anythingllm/storage/`.
- Configuración local persistida: `.runtime/anythingllm/.env`.
- Claves aleatorias generadas una sola vez: `.runtime/secrets.env`.
- Pesos de Ollama: volumen Docker `origenlm_origen_ollama` (su nombre puede variar según Compose).
- Todo `.runtime/` está excluido de Git. **Copia los datos antes de eliminar el Codespace**: los volúmenes del Codespace no son almacenamiento externo garantizado.
- No publiques el puerto 3001 sin autenticación, TLS y medidas operativas adicionales. La visibilidad Private en Ports es la adecuada para desarrollo.
- No pongas tokens ni contraseñas en GitHub o en el notebook de Kaggle.
- Las imágenes oficiales se descargan con la etiqueta `:latest`; valida nuevas versiones antes de usar en producción.
- Esta configuración es para desarrollo, no alojamiento público 24/7.

Si Docker no está disponible ejecuta `docker info`; revisa la configuración del Codespace/daemon. Los servicios no podrán iniciarse hasta que Docker funcione.

## Fuentes del upstream
- https://docs.anythingllm.com/installation-docker/local-docker
- https://docs.useanything.com/features/customization
- https://github.com/ollama/ollama
