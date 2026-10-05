# Carpeta de memes / reacciones

Con `--memes folder` (o "Mi carpeta de memes" en el Studio), los cortes de reacción
de ~2 segundos usan los archivos de esta carpeta (o de la que pases con `--memes-dir`).
Si una emoción no tiene archivos, reacciona la nutria (dibujada por la IA, o su pose).

## Cómo ordenarla

Una subcarpeta por emoción, o archivos que empiecen por la emoción:

```
resource/memes/
  sorpresa/      gato-sorprendido.png, ...
  risa/          ...
  facepalm/      ...
  mente/         (cabeza que explota, "mind blown")
  confundido/
  miedo/
  triste/
  orgullo/
  sospecha/
  panico/
  risa-perro.mp4         <- también vale: "<emoción>-lo-que-sea.ext"
```

Nombres aceptados (español o inglés): `sorpresa`/`shock`, `mente`/`explota`/`mindblown`,
`risa`/`laugh`, `facepalm`/`verguenza`, `confundido`/`confused`, `miedo`/`scared`,
`triste`/`sad`, `orgullo`/`proud`, `sospecha`/`suspicious`, `panico`/`panic`.

Formatos: PNG, JPG, WEBP (imágenes) y MP4, WEBM, MOV, GIF (se reproducen en bucle, sin sonido).

## Derechos de autor (importante para monetizar)

Casi todos los memes son fotogramas de películas, series o fotos con dueño. YouTube puede
reclamarlos o considerar el video "contenido reutilizado". Usa solo material que puedas usar:
memes que hagas tú, imágenes con licencia libre (CC0, Pexels, Pixabay) o plantillas que hayas
comprado. La opción `--memes otter` es la más segura: la reacción es de tu propia mascota.

Los archivos que pongas aquí no se suben a git (ver `.gitignore`).
