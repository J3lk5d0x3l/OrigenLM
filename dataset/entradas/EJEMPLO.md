# EJEMPLO (este archivo no se procesa)

Copiá este contenido a un archivo nuevo terminado en `.txt` y editalo.
Este de acá queda solo como referencia: el script ignora los archivos que
empiezan con "ejemplo".

```
## PREGUNTA
¿Cómo veo qué archivos cambié antes de guardar?

## RESPUESTA
Con `git status`. Te muestra los archivos modificados, los nuevos sin seguimiento
y los que ya están listos para el commit. Si querés ver el detalle línea por
línea, usá `git diff`.

## PREGUNTA
Me equivoqué en el mensaje del último commit, ¿se puede arreglar?

## RESPUESTA
Sí, siempre que no lo hayas subido todavía: `git commit --amend -m "mensaje nuevo"`
reescribe el último commit. Si ya lo subiste, es mejor no reescribir historia
compartida y dejar el arreglo en un commit nuevo.

P: ¿Qué diferencia hay entre `git fetch` y `git pull`?
R: `git fetch` baja los cambios pero no toca tu rama; podés revisarlos antes.
`git pull` es `fetch` + `merge` (o rebase), así que modifica tu trabajo de una vez.
```
