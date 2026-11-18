# Cómo agregar tus propias respuestas

Todo lo que escribas acá se convierte en datos de entrenamiento para OrigenLM.
No hace falta saber JSON: es texto plano.

## 1. Creá un archivo

Dentro de esta carpeta (`dataset/entradas/`), creá un archivo nuevo con
extensión `.txt`. Por ejemplo `dataset/entradas/mis_respuestas.txt`.

Se pueden tener varios archivos; se leen todos menos `LEEME.md` y `EJEMPLO.md`.

## 2. Escribí los pares

Cada par es una **pregunta** y una **respuesta**, con este formato:

```
## PREGUNTA
¿Qué hace un `git commit`?

## RESPUESTA
Guarda una foto del estado del repositorio. A partir de ese punto podés volver
atrás, comparar o compartir los cambios.
```

El texto puede ocupar varias líneas: sigue leyendo hasta el próximo marcador.

También sirve la forma corta:

```
P: ¿Qué es una rama en Git?
R: Una línea de trabajo paralela; sirve para probar cambios sin romper main.
```

## 3. Cargalos y verificá

```bash
python scripts/agregar_dialogos.py
```

Muestra cuántos pares nuevos entraron y cuántos se descartaron por estar
repetidos. Después regenerás el dataset de fine-tuning:

```bash
python scripts/preparar_sft.py
```

## Reglas prácticas

- **Una idea por respuesta.** Mejor 5 respuestas cortas que una gigante.
- **Escribí como hablás.** El modelo copia el tono; si escribís seco, responde seco.
- **Nada de datos personales** (direcciones, teléfonos, claves, correos ajenos).
- **No inventes:** si la respuesta es sobre código, que sea correcta. El modelo
  aprende también los errores.
- Los duplicados se descartan solos, podés repetir sin miedo.

## Qué esperar

Con cientos de pares el modelo aprende **estilo y formato**. Para que además
sepa *de qué* habla hacen falta miles de ejemplos, y ahí conviene sumar datasets
públicos en español además de los tuyos.
