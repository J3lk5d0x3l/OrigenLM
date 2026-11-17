# Fase de datos V3 y evaluación independiente

## 1. Punto de partida medido

| | Valor |
|---|---|
| Corpus actual | 116.797.208 tokens (Wikipedia es, 52.512 documentos) |
| Entrenamiento / validación | 111.967.457 / 4.829.751 tokens |
| Caracteres por token | 3,60 |
| Parámetros del modelo | 6.819.072 |

Regla de Chinchilla (≈20 tokens por parámetro) para 6,8M de parámetros:
**≈136M tokens**. Ya tenemos 112M: el modelo está cerca de su punto óptimo de
cómputo. Añadir 500M de tokens al mismo modelo rinde muy poco; para aprovechar
más datos hay que subir parámetros, y eso exige GPU.

## 2. Mezcla propuesta (no descargar nada sin autorización)

| Fuente | Cuota | Licencia declarada | Estado |
|---|---|---|---|
| Wikipedia es (`wikimedia/wikipedia`, `20231101.es`) | 50% | CC BY-SA 4.0 / GFDL | **verificada** en `dataset/fuentes_es/procedencia.json` |
| FineWeb-2 `spa_Latn` | 25% | ODC-By 1.0 | verificar ficha antes de usar |
| Gutenberg en español | 5% | dominio público (texto); la marca Gutenberg tiene condiciones aparte | verificar |
| OASST2, subconjunto español | 10% | Apache-2.0 (declarada) | verificar |
| Dolly-15k traducido al español | 5% | CC BY-SA 3.0 (original en inglés) | obra derivada: conservar atribución |
| Código con licencia permisiva (MIT/Apache-2.0) | 5% | por repositorio | verificar uno por uno |

Reglas que no se negocian:

1. Ningún dataset entra sin un `procedencia.json` con URL, licencia, fecha,
   SHA-256 y número de documentos (el mismo formato que ya usa el corpus).
2. Deduplicación por SHA-256 entre **todas** las fuentes, no solo dentro de una.
3. El texto de validación se separa por hash de documento, nunca por posición.
4. Filtro de idioma barato y explícito (`parece_espanol` ya existe) y descarte
   de páginas dominadas por tablas.

## 3. Evaluación independiente

`dataset/preparado_v2_es/val.bin` mide **modelado de lenguaje**, no capacidad de
asistente. Para no engañarnos hace falta un set aparte:

- `evals/prompts_es.jsonl` — 20 indicaciones escritas a mano, fuera del corpus,
  con comprobaciones deterministas (número exacto, subcadenas, mínimo de
  palabras). Greedy (`temperature=0`) para que dos ejecuciones sean comparables.
- `scripts/evaluar_modelo.py --salida evals/informe.json` — guarda NLL,
  perplejidad, paso, hash del checkpoint y cada respuesta.
- Pendiente de añadir: 200 preguntas de estilo examen revisadas a mano y
  FLORES-200 español para traducción (verificar licencia antes de descargar).

El set de evaluación **no se entrena** ni se usa para elegir hiperparámetros más
de una vez por experimento; si se toca, se documenta.

## 4. "Meter muchos bins" no agranda el modelo

Los `.bin` son datos, no parámetros. El tamaño del modelo sale de
`configs/*.json` (`dim`, `n_layers`, `hidden_dim`), no de cuántos bins se
concatenen. Añadir bins hace que el mismo modelo vea más texto; sin más
parámetros, el beneficio se agota.

Proyección (aritmética, no medición) usando los 1.900–2.000 tok/s medidos en
este Codespace de 4 núcleos, suponiendo coste lineal en parámetros:

| Parámetros | Tokens óptimos (20×) | 1 época de 112M tokens en CPU | ¿Tiene sentido aquí? |
|---|---|---|---|
| 6,8M (actual) | 136M | ~16 h | sí |
| 27M | 540M | ~62 h | solo con GPU o por tandas |
| 68M | 1.360M | ~155 h | no en CPU |

Nada de esto se acerca a un modelo de cientos de miles de millones de
parámetros: la diferencia es de órdenes de magnitud en cómputo, no de
configuración.

## 5. Licencia y atribución del corpus actual

El corpus es una obra derivada de Wikipedia en español. Cualquier
redistribución (incluido el modelo publicado con sus datos) debe conservar la
atribución y compartir igual (CC BY-SA 4.0). El aviso ya está en
`dataset/fuentes_es/procedencia.json` y en `dist/MANIFIESTO.md` al exportar.
