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
- **Pruebas:** `.venv/bin/python -X utf8 -m pytest -q test` (unas 1527 pruebas, tardan unos 5 minutos). La cobertura mínima que exige el CI es del 70 %.
- **Lint:** `.venv/bin/ruff check app cli.py list_video.py research.py voice_lab.py main.py webui studio test docs/skill`
- **Estilo del código:**
  - Seguir el estilo de cada archivo: docstrings cortos en inglés y comentarios solo donde aportan.
  - Mensajes de commit en inglés.
  - Cada cambio debe llevar sus pruebas.

## Cómo se usa

- **Interfaz gráfica, "Cabeceando Studio":** `./studio.sh` abre http://127.0.0.1:8600. Con `./studio.sh --install` se agrega al menú de aplicaciones.
  - Páginas: Proyectos, Guion, Voz, Estilo, Producir, Resultados, Plan de edición, Investigación y Ajustes.
  - Los renders corren en segundo plano y se ve su progreso.
- **Desde la terminal:** `python list_video.py ...`. Ejemplo:
  ```
  uv run python list_video.py --subject "Por qué los call centers están desapareciendo" --format story \
    --items 6 --video-language es-CO --look doodle --logo nutria \
    --voice-name gemini:Schedar-Even --voice-style calmado --assets nutria
  ```
  - Con `--script-only` solo escribe el guion (JSON) para revisarlo; luego se renderiza con `--script archivo.json`.
- **Laboratorio de voces:** `python voice_lab.py` compara varias voces con el mismo texto.
- **Investigación de canales:** `python research.py --channel @Canal --analyze`. Usa la YouTube Data API y calcula qué videos destacan sobre el promedio del canal.
- **Resultados:** quedan en `storage/tasks/<id>/` (`final-1.mp4`, `chapters.txt`, `credits.txt`, `edit-plan.json`).

## Configuración clave (`config.toml`)

- **LLM y Vertex AI:** `llm_provider = "gemini"`, `gemini_use_vertexai = true`, `gemini_vertex_project = "<mi proyecto>"`, `gemini_vertex_location`.
- **Voz Gemini:**
  - Modelo: `gemini_tts_model`. Recomendado: `gemini-2.5-pro-preview-tts`.
  - Estilo: `gemini_tts_style`. Acepta los presets divulgador, entusiasta, profe, calmado, sereno y narrador, o un texto propio.
- **Imágenes:** `gemini_vision_model` (revisa las imágenes, por defecto gemini-2.5-flash) y `gemini_image_model` (dibuja, por defecto imagen-4.0-fast-generate-001).
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

## Tipos de escena y de toma disponibles

- **Escenas clásicas:** statement, stat, sequence, compare, diagram, figure, zoom, story, steps, bars, grid, formula, timeline, gauge, question.
- **Escenas científicas:** definition, equation, annotate, chain, branch.
- **Tomas del estilo dibujado:** single, speech, illustration.
- **Uso interno:** opener (portada de sección).

## Pendiente o ideas para seguir

- Probar en mi PC el estilo dibujado con Gemini real y ajustar el prompt de dibujo (`DOODLE_PROMPT` y `SCENE_PROMPT` en `gemini_media.py`) hasta que se parezca a Cápsula.
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
