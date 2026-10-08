# Contexto para continuar: herramienta de videos del canal "Cabeceando"

Pega este texto al inicio de un chat nuevo (o pide al asistente que lea este archivo) para continuar el trabajo.

## Quién soy y qué busco

- Soy estudiante de medicina y programador en Colombia (hablo español y portugués).
- Tengo un canal educativo de YouTube sin cara, llamado **"Cabeceando"**. La mascota es una **nutria con gafas redondas, suéter verde azulado y un lápiz en la oreja**, y el lema es "X explicada para nutrias".
- Quiero la mejor herramienta automática de videos educativos: profesional, muy gráfica, que no parezca hecha con IA y que me permita monetizar.
- **El estilo que quiero es el de "Cápsula Mental":**
  - todo dibujado en caricatura sobre un fondo naranja-amarillo, sin videos de stock;
  - los dibujos aparecen a medida que el narrador los nombra;
  - voz calmada, con muchas pausas e ideas bien terminadas;
  - etiquetas escritas a mano en mayúsculas.
- **Para los guiones me inspiran canales como "Cómo es ser cada rango de NCIS realmente":** abren en una situación atrapante y cuentan una historia continua, sin empezar de forma técnica.
- Uso **Gemini por Vertex AI** (credenciales de `gcloud auth application-default login`).
- Mi voz favorita hasta ahora es **`gemini:Schedar-Even` con el estilo "calmado"**. ElevenLabs sigue disponible como opción (los créditos son mensuales).
- Mi PC: Fedora o CachyOS, repositorio en `/home/william/Desktop/Repos/MoneyPrinterTurbo`, ffmpeg en `/usr/bin/ffmpeg`.

## Repositorio y forma de trabajar

- **Repositorio y rama:**
  - Repositorio: `cosmicalotter/MoneyPrinterTurbo`, un fork de MoneyPrinterTurbo.
  - Rama de trabajo: **`claude/relaxed-sagan-nl9ga3`**. Todo se commitea y se sube ahí.
  - No crear pull requests salvo que yo lo pida.
- **Entorno:** Python 3.11 con `uv`, entorno virtual en `.venv`. Se instala con `uv sync --frozen`.
- **Pruebas:** `.venv/bin/python -X utf8 -m pytest -q test` (unas 1610 pruebas, tardan unos 6 minutos). La cobertura mínima que exige el CI es del 70 %.
- **Lint:** `.venv/bin/ruff check app cli.py list_video.py research.py voice_lab.py main.py webui studio test docs/skill`
- **Estilo del código:**
  - Seguir el estilo de cada archivo: docstrings cortos en inglés y comentarios solo donde aportan.
  - Mensajes de commit en inglés.
  - Cada cambio debe llevar sus pruebas.

## Cómo se usa

- **Interfaz gráfica, "Cabeceando Studio":** `./studio.sh` abre http://127.0.0.1:8600. Con `./studio.sh --install` se agrega al menú de aplicaciones.
  - Páginas: Proyectos, Guion, Voz, Estilo, Producir, Revisar, Resultados, Plan de edición, Investigación y Ajustes.
  - Los renders corren en segundo plano y se ve su progreso.
- **Desde la terminal:** `python list_video.py ...`. Ejemplo:
  ```
  uv run python list_video.py --subject "Por qué los call centers están desapareciendo" --format story \
    --items 6 --video-language es-CO --look doodle --logo nutria \
    --voice-name gemini:Schedar-Even --voice-style calmado --assets nutria \
    --image-quality high --review
  ```
  - Con `--review` se detiene antes de montar el video: se revisan las imágenes en **Revisar** del Studio y luego se renderiza con `--edit-plan storage/tasks/<id>/edit-plan.json --task-id <id>`.
  - Con `--script-only` solo escribe el guion (JSON) para revisarlo; luego se renderiza con `--script archivo.json`.
- **Laboratorio de voces:** `python voice_lab.py` compara varias voces con el mismo texto.
- **Investigación de canales:** `python research.py --channel @Canal --analyze`. Usa la YouTube Data API y calcula qué videos destacan sobre el promedio del canal.
- **Resultados:** quedan en `storage/tasks/<id>/` (`final-1.mp4`, `chapters.txt`, `credits.txt`, `edit-plan.json`).

## Configuración clave (`config.toml`)

- **LLM y Vertex AI:** `llm_provider = "gemini"`, `gemini_use_vertexai = true`, `gemini_vertex_project = "<mi proyecto>"`, `gemini_vertex_location`.
- **Voz Gemini:**
  - Modelo: `gemini_tts_model`. Recomendado: `gemini-2.5-pro-preview-tts`.
  - Estilo: `gemini_tts_style`. Acepta los presets divulgador, entusiasta, profe, calmado, sereno y narrador, o un texto propio.
- **Imágenes:** `gemini_vision_model` (revisa las imágenes, por defecto gemini-2.5-flash), `gemini_image_model` (vacío = según `--image-quality`: gemini-3.1-flash-image o gemini-3-pro-image; Imagen 4 solo si se escribe aquí) y `gemini_video_model` (Veo; vacío = veo-3.1-fast-generate-001) con `gemini_video_location` (vacío = la región de Vertex; poner `us-central1` si Veo no responde en `global`).
- **Google Cloud TTS:** `gcloud_tts_api_key` (opcional) y `gcloud_tts_pitch`. Las voces son `gcloud:es-US-Chirp3-HD-*`.
- **ElevenLabs:** en `[elevenlabs]` van `api_key`, `model_id`, `stability`, `similarity_boost` y `style`. Las voces se escriben `elevenlabs:<voice_id>`; sin el prefijo `elevenlabs:` el sistema las trata como voces de Azure.
- **Imágenes y videos de stock:** `pexels_api_keys` y `pixabay_api_keys`.
- **Investigación:** `youtube_api_key`.

## Arquitectura (archivos principales)

| Archivo | Qué hace |
|---|---|
| `list_video.py` | CLI de videos en formato lista o historia; reenvía las opciones de `cli.py` (voz, aspecto, música, subtítulos). |
| `app/services/list_video.py` | Render: narra cada segmento (pasada 1), arma el plan, dibuja cada segmento con ffmpeg (pasada 2), mezcla efectos, masteriza la voz y une todo. |
| `app/services/list_video_editor.py` | "Editor" con IA: `EditOptions`, plan de edición o storyboard, tiempos de escenas, imágenes flotantes y presentador, búsqueda de imágenes y dibujos, lienzo, logo, píldora de capítulo, subscribe, efectos. |
| `app/services/list_video_scenes.py` | Escenas a pantalla completa, unos 23 tipos (ver más abajo), tiempos (`time_scenes`, `time_shots`), `SceneRenderer`, tipografía a mano Patrick Hand, animaciones (pop, stamp, flechas, trazo vivo / "line boil"). |
| `app/services/list_video_host.py` | La nutria presentadora: cuándo entra y sale, caras cada ~3 s, reacciones, `--host-presence`. |
| `app/services/list_video_fx.py` | Overlays de ffmpeg (modos still, frames, concat, sequence y media), stickers, recortes, chips, efectos de sonido, badge del logo, `write_ffconcat`. |
| `app/services/llm.py` | Prompts: guion en lista (`build_list_script_prompt`), guion de historia (`build_story_script_prompt`), plan de edición (`build_edit_plan_prompt`), storyboard dibujado (`build_storyboard_prompt`), paso que rellena huecos sin imagen, traducción, metadatos sociales. |
| `app/services/gemini_media.py` | Gemini como editor de imágenes (`choose_picture`: beat, scene, figure, opener, annotate; `choose_figure`; `locate_parts`) y como dibujante (`illustrate`, `illustrate_sequence`, `draw` con referencia de la mascota). |
| `app/services/voice.py` | Proveedores de voz (Edge, Azure, Gemini TTS, ElevenLabs, Fish Audio, Chatterbox, Kokoro, MiniMax…). |
| `app/services/voice_google.py` | Gemini TTS robusto (reintentos, cambio de modelo si falta, audio cortado), Google Cloud TTS (Chirp 3 HD), presets de estilo, laboratorio de voces. |
| `app/services/voice_polish.py` | Calidad de la voz: 48 kHz, cadena de masterización, pausas alargadas, alineación de frases a partir de los silencios. |
| `app/services/studio.py` + `studio/Studio.py` | Lógica y páginas de Streamlit del Studio. `studio.sh` es el lanzador. |
| `app/services/icons.py` | Iconos OpenMoji (CC BY-SA, se acredita), búsqueda y alternativas. |
| `app/services/web_images.py` | Imágenes de Wikimedia, Pexels y Pixabay con atribución. |
| `resource/memes/` | Tu carpeta de memes por emoción (`--memes folder`); su contenido no se sube a git. |
| `resource/characters/nutria/` | La nutria: SVG, script que la dibuja y `personaje/*.png` con 12 expresiones más sus versiones `_habla`. |

## Lo construido, ronda por ronda

### Ronda 1: base y mascota
- Formato de video en lista con segmentos intro, ítems y outro, cada uno narrado por separado y con capítulos para YouTube.
- Edición automática con IA: fondos de stock que siguen la narración, imágenes y datos que aparecen cuando se mencionan, efectos de sonido y volumen a -14 LUFS.
- La nutria presentadora: aparece y desaparece, reacciona y cambia de cara; mascota en SVG/PNG.
- Versiones en otros idiomas (`--also-in`), que reutilizan el plan visual.

### Ronda 2: presentador e imágenes
- Poses estáticas con transición suave y rebote, sin el destello al salir.
- Revisión de cada imagen con Gemini vision.
- Iconos OpenMoji, fuente Patrick Hand, ilustraciones de IA.
- Primeras escenas explicativas.

### Ronda 3: más imágenes y escenas
- Imágenes centradas o en grupos simétricos de 2 a 4.
- Figuras a pantalla completa, cuyo tiempo en pantalla decide Gemini.
- Recortes seguros de stickers.
- 15 tipos de escena con imágenes reales revisadas.
- La cara de la nutria cambia cada ~3 s.
- Sin barra de progreso por defecto.

### Ronda 4: más científico y la interfaz
- Efectos de sonido al 65 % (`--sfx-volume`) y nutria menos presente (`--host-presence low`).
- La píldora del capítulo se esconde durante las escenas.
- Portada de cada sección: número, título y una imagen estrictamente del título (`--no-openers`).
- Guion más científico: define términos, da unidades y fórmulas y explica el porqué.
- Escenas nuevas: `definition`, `equation`, `annotate` (imagen real con flechas que Gemini ubica), `chain` y `branch`.
- Paso que rellena con imágenes las frases que solo tenían video de fondo.
- Gemini TTS robusto, Google Cloud TTS y `voice_lab.py`.
- Cabeceando Studio (Streamlit) con su lanzador para Linux.

### Ronda 5: estilo Cápsula, historias y calidad de voz
- **Voz:**
  - Se mantiene a 48 kHz en todo el proceso (antes bajaba a 24 kHz).
  - Se quitó la saturación que el mezclador aplicaba a toda la voz.
  - Masterización tipo estudio (`--no-voice-polish` la desactiva).
  - El volumen se normaliza en dos pasadas y el audio sale en AAC a 256k.
  - Pausas entre frases (`--pause 0.5`) y alineación de frases por silencios, para que las imágenes caigan en la palabra correcta.
  - Ajustes de ElevenLabs configurables y nuevo preset "sereno".
- **`--look doodle` (estilo Cápsula):**
  - Todo dibujado sobre un lienzo de color (`--canvas-color`, por defecto `#F4C24F`).
  - La IA hace un storyboard con una composición nueva cada 1–2 frases. Tipos de toma: `single`, `speech` (globo de diálogo), `illustration` (escena 16:9 completa con texto grande), `stat`, `bars`, `sequence` con tachones, etc.
  - Dibujos de Gemini/Imagen en estilo tinta (`--max-drawings`); la nutria se dibuja a partir de su propia imagen; si no hay dibujo, se usan iconos.
  - Los dibujos tiemblan levemente como animación a mano (`--no-boil` lo apaga).
  - Logo en la esquina (`--logo nutria`) y sin videos de stock.
- **`--format story`:** una historia continua que abre en una situación atrapante, sin números, títulos ni portadas en pantalla.
- **Imágenes flotantes mucho más grandes**, y sin repetir imágenes, iconos ni dibujos.
- **El Studio** tiene por defecto el estilo dibujado, el formato historia, Schedar-Even con "calmado", pausa de 0,5 s y masterización activada.

### Ronda 6: estilo animatic, videos reales enmarcados, reacciones y dibujos más sólidos
- **Diagnóstico del video de prueba ("La electricidad explicada"):** solo la primera escena fue una ilustración de IA; el resto fueron íconos OpenMoji sueltos (a veces equivocados: 😘 para "onda de energía", 🀄 para "primer contacto"), tomas de hasta 21 s sin cambios y comparaciones con la mitad vacía durante ~12 s.
- **Ritmo de animatic (`--shot-seconds`, por defecto 3):** el storyboard cambia de imagen cada 2–4 s. A cada segmento se le dice cuánto dura y cuántas tomas necesita; los videos largos se planifican por grupos de segmentos en paralelo; un segundo pase rellena los tramos donde una imagen se quedaría demasiado.
- **Ilustraciones a pantalla completa como base (~55 %):** estilo caricatura animada pulida (`--drawing-style cartoon`; `ink` mantiene el garabato a tinta), con movimiento de cámara (acercar, alejar, paneo izquierda/derecha) y fundido entre ellas. Una ilustración con `"continue": true` se redibuja a partir de la anterior cambiando solo una cosa: así se cuenta una acción como fotogramas de animación. Los subtítulos grandes van en el tercio inferior.
- **Composiciones explicativas más trabajadas:** el dibujo único aparece "dibujándose" sobre una mancha de papel; la comparación ya no muestra la línea divisoria vacía; el primer elemento de una composición se ve desde el inicio.
- **Videos reales dentro del dibujo (`--clips none|some|more`):** clips de Pexels (luego Pixabay) en un marco a tinta con cinta adhesiva sobre el fondo, o a pantalla completa; Gemini revisa un fotograma; si no hay clip, se dibuja la escena; se acreditan en `credits.txt`.
- **Reacciones tipo meme (`--memes off|otter|folder`, por defecto off):** cortes de ~2 s en los remates, como mucho uno cada 40 s, con rayos de cómic y un "boom" suave. `otter` dibuja a la nutria reaccionando (sin problemas de derechos); `folder` usa tus imágenes/videos por emoción en `resource/memes/` o `--memes-dir` (ver su README sobre derechos de autor). Después del meme vuelve la imagen que interrumpió.
- **Dibujo más robusto:** si Imagen falla (región, proyecto o prompt filtrado) dibuja `gemini-2.5-flash-image` y no se vuelve a pedir a Imagen; los modelos ocupados (429/503) se reintentan; las advertencias del render dicen cuántos dibujos fallaron y por qué, y si se acabó `--max-drawings`.
- **Íconos solo como respaldo, y revisados:** Gemini elige entre varios candidatos el que de verdad representa la cosa; si ninguno sirve, se muestra solo la etiqueta.
- **Studio:** en Estilo → dibujado: "Cambio de imagen cada (s)", "Videos reales dentro del dibujo", "Estilo de dibujo" y "Reacciones tipo meme".

### Ronda 7: animaciones cortas con IA, cámara suave y nunca solo texto
- **Diagnóstico del segundo video ("La electricidad explicada", 3:20):** las ilustraciones que salieron se veían bien, pero eran imágenes sueltas de 5–9 s; muchos dibujos fallaron y el respaldo dejó tomas de solo texto ("MEDICIÓN EN VOLTIOS" 13 s, "PRESIÓN DISPONIBLE" 12 s) o el fondo vacío hasta 11 s; el color de marca (casi igual al fondo) volvió invisible el texto de las definiciones; la primera ilustración traía bordes blancos; el zoom temblaba; los clips de tormenta eran casi negros.
- **Animaciones (`"type": "animation"`):** 2–4 dibujos de una misma acción, ~1 s cada uno; cada dibujo se redibuja desde el anterior (mismo lugar y personajes, cambia una cosa) y se funde con el siguiente bajo un solo movimiento de cámara. Son ~50 % de las tomas; ~35 % ilustraciones; composiciones como mucho ~12 %.
- **Cámara suave:** cada fotograma se recorta con precisión sub-píxel y aceleración suave al inicio y al final (adiós al temblor del zoompan de ffmpeg); movimientos más pequeños y lentos (zoom 1,00→1,06, paneos cortos). Se recorta la hoja blanca alrededor de un dibujo.
- **Composiciones (fondo naranja):** pocas, de al menos 4 s (la toma siguiente espera hasta 2 s), y **nunca solo texto**: elementos sin imagen se quitan, una etiqueta sola lleva a la nutria, una fórmula la presenta la nutria; si una composición no puede mostrar imágenes, se omite.
- **Imágenes en vez de íconos:** cada elemento prueba dibujo IA → imagen real (PNG recortado si se puede) que Gemini aprueba → ícono revisado. **Auditoría:** Gemini revisa cada dibujo (que muestre lo pedido, sin texto ni deformaciones, la nutria reconocible) y lo rehace una vez si falla.
- **Fiabilidad:** máximo 2 dibujos a la vez; ante cuota agotada (429) espera 5, 15, 30 y 60 s; los dibujos que aún fallan se reintentan al final y, si no, se usa una foto real aprobada del momento (cada toma lleva un `query` en inglés para eso). Una toma que falla al renderizarse la reemplaza la nutria con su etiqueta.
- **Otros:** el texto en color de marca se oscurece si no contrasta con el fondo; se descartan clips oscuros; `render-report.txt` (también en Resultados del Studio) resume tomas, dibujos (hechos, fallidos, rehechos), respaldos y advertencias. **Sin logo** en la esquina por defecto (los proyectos guardados también lo pierden; se vuelve a activar en Estilo). `--max-drawings` por defecto 260.

### Ronda 8: documental animado, fotos históricas reales y revisión antes de renderizar
- **Diagnóstico del tercer video ("Los científicos más importantes", 3 h de render):** íconos gigantes a pantalla completa (dibujos de animación que fallaban y caían a un ícono), imágenes genéricas fuera de contexto (la nutria en un camino mientras se habla de la peste), demasiadas comparaciones partidas, ritmo muy rápido, temas repetidos (el polonio dos veces), artefactos en objetos complejos (el telescopio), la nutria de las fórmulas tapando texto y la notación `*`/`^` fea.
- **El director ahora planifica en tres pasos:** (1) una "biblia visual" que lee todo el guion: dirección de arte, personajes recurrentes con su aspecto fijo y un retrato real, y las fotos históricas reales que vale la pena mostrar por sección; (2) el guion gráfico, por grupos de segmentos en paralelo; (3) un montajista IA que lo corrige antes de dibujar (imágenes fuera de contexto, historia contada con dibujos cuando hay foto real, repetidos, ritmo). Tomas repetidas se quitan automáticamente.
- **Fotos de archivo (`"type": "archive"`):** pinturas, grabados, fotos históricas, manuscritos e instrumentos reales de Wikimedia Commons (en grande); Gemini confirma que son auténticas y del momento exacto (rechaza fotos de stock montadas, recreaciones, IA). Si ninguna pasa, el momento se dibuja. Se muestran a pantalla completa (o enteras sobre una copia desenfocada si son verticales) con un pie tipo museo ("Londres, 1665") abajo a la izquierda.
- **Mezcla:** ~30 % archivo (cuando hay historia), ~35 % animaciones, ~25 % ilustraciones, composiciones ≤ 10 %, una sola comparación como mucho. **Ritmo calmado:** imagen nueva cada 4–7 s (`--shot-seconds` 5 por defecto, de 3 a 8; los proyectos guardados con 3 pasan a 5).
- **Mejores dibujos:** modelos Gemini por calidad (`--image-quality`: economy = Gemini 2.5 Flash Image ~US$0,04; standard = Gemini 3.1 Flash Image ~US$0,07 (US$0,10 una escena a 2K); high = Gemini 3 Pro Image en escenas nuevas y fichas de personaje; max = Pro en todo ~US$0,13), escenas a 2K, estilo más simple que pide formas correctas y reconocibles ("menos detalles antes que detalles mal hechos"), estilo `flat` aún más minimalista. Cada persona recurrente tiene una **ficha de personaje** dibujada a partir de su retrato real, y todos sus dibujos la siguen. La **auditoría** compara cada dibujo con la frase que se dice en ese momento (y la anatomía y los objetos) y, si falla, se redibuja con la corrección que propone Gemini. **Nunca** un ícono gigante en lugar de un dibujo.
- **Videos con IA opcionales (`--ai-videos N`):** las N ilustraciones más largas que tienen un movimiento suave se convierten en videos Veo 3.1 de 4–8 s sin sonido (~US$0,10/s con Veo 3.1 Fast: 10 videos de 5 s ≈ US$5); se reproducen una vez, un poco más lento si la toma es larga, y se quedan en el último cuadro.
- **Secciones:** el guion dice el nombre de cada sección en voz alta ("Albert Einstein."); la sección abre con una **tarjeta de color propio** con la foto real del protagonista en un marco tipo polaroid, el número y el nombre; luego la **etiqueta "01 Albert Einstein"** queda arriba a la izquierda sobre las imágenes y se aparta en las composiciones. Aplica también al formato historia.
- **Intro:** gancho y luego una frase que dice de qué trata el video usando la idea del título (con la nutria puede bromear una vez: "...bueno, aunque yo sea una nutria").
- **Nada tapa nada:** la nutria de las fórmulas tiene su propia columna; las fórmulas se escriben como en el tablero (×, ², √); el botón de suscribirse va arriba a la derecha; las imágenes con transparencia ya no dejan esquinas negras.
- **Revisar antes de renderizar:** `--review` (o "Preparar revisión" en Producir) hace el plan y todas las imágenes y se detiene. En **Revisar** cada toma muestra sus imágenes y la frase que suena; se puede mantener, pedir otra versión (cambiando qué se dibuja), cambiar por una foto real o quitar. "Renderizar con esta revisión" usa exactamente lo que se mantuvo, con la misma narración (las narraciones quedan en caché).
- **Progreso:** mientras se preparan las imágenes, el Studio cuenta cuántas van listas (antes se quedaba en "Montando la sección 1" mucho tiempo, porque todas las imágenes del video se hacen al empezar la sección 1).

## Tipos de escena y de toma disponibles

- **Escenas clásicas:** statement, stat, sequence, compare, diagram, figure, zoom, story, steps, bars, grid, formula, timeline, gauge, question.
- **Escenas científicas:** definition, equation, annotate, chain, branch.
- **Tomas del estilo dibujado:** archive (foto o pintura histórica real con pie de museo), animation (2–4 dibujos de ~1,5–2 s que continúan uno del otro), illustration (con `continue`, `camera`, `motion` para Veo y `query` de respaldo), single, speech, clip (video real enmarcado o a pantalla completa) y meme (reacción por emoción: shock, mindblown, laugh, facepalm, confused, scared, sad, proud, suspicious, panic). Los dibujos pueden llevar `characters` (ids de la biblia visual) y, tras una revisión, `image`/`video` fijados.
- **Uso interno:** opener (portada de sección).

## Pendiente o ideas para seguir

- Probar en mi PC la ronda 8 con Gemini y Veo reales (las llamadas a los modelos están probadas con simulaciones, no con la API real): revisar en `render-report.txt` cuántas fotos de archivo se usaron y si algún modelo aparece como no disponible (región `global` recomendada para los modelos Gemini 3; Veo suele pedir `us-central1`). Ajustar los prompts de estilo (`CARTOON_PROMPT`, `CARTOON_SCENE_PROMPT`, `FLAT_*` y `NEXT_FRAME_PROMPT` en `gemini_media.py`).
- Costo aproximado por video de ~5 min con la ronda 8: menos dibujos que antes (el ritmo es más calmado y ~30 % son fotos reales gratis): ~120–180 dibujos ≈ US$8–12 en calidad estándar, ~US$15–25 en alta; + Veo si se activa. Todo queda en caché; re-renderizar tras una revisión solo cobra lo que se rehace.
- Después de cada render, revisar `render-report.txt` (o el informe en Resultados): si dice muchos "failed", copiar la línea "Last drawing error" para diagnosticar (cuota de Vertex, región, filtro de seguridad).
- Escuchar la voz masterizada y ajustar la cadena (`polish_chain` en `voice_polish.py`) y la pausa.
- Ideas:
  - una voz propia del canal con voz clonada (ElevenLabs o Fish Audio);
  - animaciones de Wikimedia (GIF/webm con licencia libre) para el estilo con videos de stock;
  - más tipos de toma (mapa, calendario, comparación de tamaños);
  - un formato vertical para Shorts.
- Evitar GIFs de Giphy y Tenor: casi todos tienen derechos de autor y arriesgan reclamos de Content ID o la desmonetización.

## Problemas ya resueltos (por si reaparecen)

- **"moonshot api_key not set":** falta configurar el LLM. Usar `llm_provider = "gemini"` con Vertex AI.
- **"Invalid voice 'SzWo…'" de Azure:** a la voz de ElevenLabs le falta el prefijo `elevenlabs:`.
- **"No es un guion válido… items.0.name":** había una sección vacía. El Studio ahora ignora las secciones vacías y explica en español qué falta.
