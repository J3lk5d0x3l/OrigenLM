# OrigenLM — hoja de ruta de conocimiento técnico

## Objetivo

Que OrigenLM avance de un modelo experimental con 133 conversaciones a un
asistente programador generalista con una biblioteca entrenable y herramientas
externas. No se promete superar a ningún modelo comercial: primero hay que
medir comprensión, exactitud y utilidad con evaluaciones reproducibles.

**Importante:** un índice de fuentes no equivale a pesos entrenados. GitHub
Actions valida el código; no realiza por sí solo un preentrenamiento masivo.

## Competencias de programación (con conjuntos separados de datos y pruebas)

- Fundamentos: algoritmos, estructuras de datos, complejidad, pruebas y depuración.
- Lenguajes: Python, JavaScript, TypeScript, Java, C, C++, Rust, Go, C#,
  Kotlin, Swift, PHP, Ruby, Lua, SQL, Bash, PowerShell, R, HTML y CSS.
- Ingeniería: Git/GitHub, CI/CD, contenedores, Linux, Windows, APIs,
  microservicios, bases de datos SQL y NoSQL, sistemas distribuidos y rendimiento.
- Datos e IA: matemáticas, estadística, aprendizaje automático, PyTorch,
  tokenización, LLM, evaluación y operaciones de modelos.
- Redes: HTTP, DNS, TCP/IP, TLS, proxy, cloud, autenticación, infraestructura.
- Ciberseguridad autorizada: desarrollo seguro, revisión de dependencias,
  criptografía aplicada, hardening, modelado de amenazas, detección, respuesta
  a incidentes, análisis de logs, laboratorios CTF y pentesting **con permiso**.
- Seguridad de aplicaciones: OWASP Top 10, mitigaciones, fuzzing controlado,
  pruebas en laboratorio, reporting y remediación.

## Biblioteca de fuentes: requisitos para el agente

1. Usa documentación oficial, manuales con licencia verificable, código
   con permiso para su uso, material original y corpus públicos con licencias
   compatibles. Registra URL, versión, licencia, atribución, idioma y SHA256.
2. Antes de descargar, comprueba las condiciones de uso y el tamaño total.
   No importes secretos, dumps privados, repositorios ajenos no autorizados,
   payloads maliciosos ni material protegido sin derechos de reutilización.
3. Deduplica documentos, conserva fragmentos de código y referencias, retira
   credenciales y divide por documento antes de tokenizar.
4. Mantén separación entre preentrenamiento de lenguaje, ajuste de
   instrucciones, ejercicios ejecutables y evaluación reservada.
5. Genera tareas verificables con soluciones y tests, no solo respuestas
   memorizadas. Mide exactitud en programación, razonamiento, seguimiento
   de instrucciones y seguridad defensiva.
6. Para consultas que requieran documentación actual, usa búsqueda documental
   (RAG) y herramientas autorizadas: no asumas que el modelo sabe datos nuevos
   por haber sincronizado GitHub.
7. Añade ejecución de código únicamente en un entorno aislado, con CPU/RAM/
   tiempo/red y permisos delimitados. Nunca ejecutes comandos sugeridos por
   un modelo directamente en el host ni permitas acceso automático a secretos.
8. Entrena en Codespaces CPU con checkpoints y cuotas revisadas. No actives
   costes externos ni usos abusivos sin autorización.

## Orden de implementación

- Primera entrega: instalación y CLI, GitHub Actions, diagnóstico de CPU,
  validador de fuentes y biblioteca local de texto.
- Segunda: pipeline de corpus por streaming, limpiador, deduplicación,
  registro de licencias, partición por documento y tokenizador BPE propio.
- Tercera: preentrenamiento real, calidad de datos, checkpoints reanudables,
  NLL/perplejidad y evaluación de programación.
- Cuarta: ajuste conversacional de código, herramientas de lectura de archivos,
  búsqueda local y ejecución aislada supervisada.
- Quinta: seguridad defensiva, pentesting autorizado en laboratorios,
  herramientas auditables y benchmarks que distingan capacidades reales.

**Estado actual:** arquitectura v2 y chat optimizado; biblioteca generalista
y entrenamiento extenso siguen pendientes. No atribuir esos conocimientos
al checkpoint de 300 pasos ni al v2 sin evidencia.
