"""
Cabeceando Studio: a desktop app (in the browser) for the whole list-video
workflow: projects, AI scripts, voices, style, renders in the background,
results with the YouTube description, edit plans, channel research and
settings. Start it with ./studio.sh (Linux) or:

    .venv/bin/python -m streamlit run studio/Studio.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.services import studio  # noqa: E402

LOGO = ROOT / "resource" / "characters" / "nutria" / "personaje" / "saludando.png"
HAND_FONT = ROOT / "resource" / "fonts" / "PatrickHand-Regular.ttf"
LANGUAGES = ["es-CO", "es-MX", "es-ES", "es-US", "en-US", "pt-BR", "fr-FR", "de-DE", "it-IT"]
STATE_LABELS = {"running": "⏳ En curso", "done": "✅ Listo", "failed": "❌ Falló", "cancelled": "⏹️ Cancelado"}

st.set_page_config(page_title="Cabeceando Studio", page_icon="🦦", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def _hand_font_css() -> str:
    """The channel's hand font embedded in the page, so it works offline."""
    if not HAND_FONT.is_file():
        return ""
    data = base64.b64encode(HAND_FONT.read_bytes()).decode()
    return f"@font-face {{ font-family: 'Patrick Hand'; src: url(data:font/ttf;base64,{data}) format('truetype'); }}"


st.markdown(
    "<style>" + _hand_font_css() + """
html, body, [class*="css"] { font-family: 'Inter', 'Source Sans Pro', sans-serif; }
h1, h2, h3 { font-family: 'Patrick Hand', cursive !important; letter-spacing: .5px; }
h1 { font-size: 2.6rem !important; }
.block-container { padding-top: 1.6rem; max-width: 1300px; }
.cab-card { border: 1px solid rgba(128,128,128,.25); border-radius: 18px; padding: 1rem 1.2rem; margin-bottom: .8rem;
            background: rgba(255,79,94,.04); }
.cab-card h4 { margin: 0 0 .3rem 0; font-family: 'Patrick Hand', cursive; font-size: 1.5rem; }
.cab-muted { opacity: .7; font-size: .9rem; }
.cab-pill { display: inline-block; padding: .1rem .6rem; border-radius: 999px; background: #FF4F5E; color: white;
            font-size: .8rem; margin-right: .3rem; }
div[data-testid="stMetricValue"] { font-family: 'Patrick Hand', cursive; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


def current_project() -> str:
    return st.session_state.get("project", "")


def require_project() -> str:
    slug = current_project()
    if not slug:
        st.info("Elige o crea un proyecto en **Proyectos** (barra lateral) para continuar.")
        st.stop()
    return slug


def settings_for(slug: str) -> studio.RenderSettings:
    key = f"settings::{slug}"
    if key not in st.session_state:
        st.session_state[key] = studio.load_settings(slug)
    return st.session_state[key]


def save_settings(slug: str, settings: studio.RenderSettings) -> None:
    st.session_state[f"settings::{slug}"] = settings
    studio.save_settings(slug, settings)


def when(timestamp) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%d/%m %H:%M") if timestamp else "—"


def sidebar() -> None:
    with st.sidebar:
        if LOGO.is_file():
            st.image(str(LOGO), width=110)
        st.markdown("## Cabeceando Studio")
        st.caption("Videos educativos explicados para nutrias 🦦")
        projects = studio.list_projects()
        slugs = [p["slug"] for p in projects]
        titles = {p["slug"]: p["title"] for p in projects}
        if slugs:
            chosen = current_project() if current_project() in slugs else slugs[0]
            st.session_state["project"] = st.selectbox(
                "Proyecto", slugs, index=slugs.index(chosen), format_func=lambda s: titles.get(s, s)
            )
        running = [j for j in studio.list_jobs(limit=10) if j["state"] == "running"]
        if running:
            st.markdown("**En curso**")
            for job in running:
                st.progress(job["progress"], text=f"{job['label'] or job['id']}: {job['stage']}")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_projects() -> None:
    st.title("Proyectos")
    left, right = st.columns([3, 2], gap="large")
    with right:
        st.markdown("### Nuevo proyecto")
        with st.form("new-project", clear_on_submit=True):
            title = st.text_input("Título de trabajo", placeholder="La electricidad explicada para nutrias")
            if st.form_submit_button("Crear proyecto", type="primary", use_container_width=True) and title.strip():
                st.session_state["project"] = studio.create_project(title)
                st.success("Proyecto creado. Sigue con **Guion**.")
                st.rerun()
        st.markdown("### Estado del sistema")
        for name, ok, detail in studio.diagnostics():
            st.markdown(f"{'✅' if ok else '⚠️'} **{name}** <span class='cab-muted'>{detail}</span>", unsafe_allow_html=True)
        st.caption("Corrige lo que falte en **Ajustes**.")
    with left:
        st.markdown("### Tus proyectos")
        projects = studio.list_projects()
        if not projects:
            st.info("Aún no hay proyectos. Crea el primero a la derecha.")
        for project in projects:
            with st.container():
                st.markdown(
                    f"<div class='cab-card'><h4>{project['title']}</h4>"
                    f"<span class='cab-pill'>{project['renders']} renders</span>"
                    f"<span class='cab-muted'>{'con guion' if project['has_script'] else 'sin guion'} · "
                    f"editado {when(project['updated'])}</span></div>",
                    unsafe_allow_html=True,
                )
                if st.button("Abrir", key=f"open-{project['slug']}"):
                    st.session_state["project"] = project["slug"]
                    st.rerun()


def _script_from_state(slug: str) -> dict:
    return st.session_state.setdefault(f"script::{slug}", studio.load_script(slug) or {
        "title": "", "intro": "", "intro_image_term": "", "items": [{"name": "", "text": "", "image_term": ""}],
        "outro": "", "outro_image_term": "",
    })


def page_script() -> None:
    slug = require_project()
    settings = settings_for(slug)
    st.title("Guion")
    st.caption(f"Proyecto: **{studio.project_meta(slug).get('title', slug)}**")
    write, upload = st.tabs(["✍️ Escribir con IA", "📥 Importar JSON"])
    with write:
        with st.form("ai-script"):
            subject = st.text_area(
                "Tema del video", placeholder="La electricidad explicada para nutrias: carga, voltaje, corriente...",
                help="El LLM escribe un guion científico pero fácil: define términos, unidades, fórmulas y el porqué.",
            )
            script_format = st.radio(
                "Formato", studio.FORMATS, index=studio.FORMATS.index(settings.script_format) if settings.script_format in studio.FORMATS else 0,
                horizontal=True, format_func=lambda v: {"story": "Historia continua (gancho y capítulos)", "list": "Lista por secciones numeradas"}[v],
                help="Historia: empieza en una situación atrapante y avanza sin cortes; los capítulos quedan solo en la descripción.",
            )
            c1, c2, c3 = st.columns(3)
            items = c1.number_input("Capítulos o secciones", 2, 30, int(settings.items))
            words = c2.number_input("Palabras por capítulo", 60, 400, int(settings.words_per_item), step=10)
            language = c3.selectbox("Idioma", LANGUAGES, index=LANGUAGES.index(settings.language) if settings.language in LANGUAGES else 0)
            if st.form_submit_button("Escribir guion", type="primary") and subject.strip():
                settings.items, settings.words_per_item, settings.language = int(items), int(words), language
                settings.script_format = script_format
                save_settings(slug, settings)
                argv = studio.build_argv(settings, subject=subject.strip(), script_only=True, output=str(studio.script_path(slug)))
                st.session_state[f"script-job::{slug}"] = studio.start_job(argv, label="Guion con IA", project=slug)
        job_id = st.session_state.get(f"script-job::{slug}")
        if job_id:
            script_job(slug, job_id)
    with upload:
        uploaded = st.file_uploader("Guion .json", type=["json"])
        if uploaded is not None:
            try:
                data = json.loads(uploaded.getvalue().decode("utf-8-sig"))
                studio.save_script(slug, data)
                st.session_state.pop(f"script::{slug}", None)
                st.success("Guion importado.")
            except Exception as exc:
                st.error(f"No es un guion válido: {exc}")

    script = _script_from_state(slug)
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Secciones", len(script.get("items") or []))
    m2.metric("Palabras", studio.word_count(script))
    m3.metric("Duración estimada", f"{studio.estimated_minutes(script)} min")
    script["title"] = st.text_input("Título del video", script.get("title", ""))
    script["intro"] = st.text_area("Intro (gancho)", script.get("intro", ""), height=100)
    items = script.setdefault("items", [])
    for number, item in enumerate(list(items)):
        with st.expander(f"{number + 1:02d} · {item.get('name') or 'Sección sin nombre'}", expanded=False):
            item["name"] = st.text_input("Nombre", item.get("name", ""), key=f"name-{slug}-{number}")
            item["text"] = st.text_area("Narración", item.get("text", ""), height=180, key=f"text-{slug}-{number}")
            item["image_term"] = st.text_input("Imagen de fondo (inglés)", item.get("image_term", ""), key=f"term-{slug}-{number}")
            b1, b2, b3 = st.columns(3)
            if b1.button("⬆️ Subir", key=f"up-{slug}-{number}", disabled=number == 0):
                items[number - 1], items[number] = items[number], items[number - 1]
                st.rerun()
            if b2.button("⬇️ Bajar", key=f"down-{slug}-{number}", disabled=number == len(items) - 1):
                items[number + 1], items[number] = items[number], items[number + 1]
                st.rerun()
            if b3.button("🗑️ Quitar", key=f"del-{slug}-{number}", disabled=len(items) == 1):
                items.pop(number)
                st.rerun()
    if st.button("➕ Añadir sección"):
        items.append({"name": "", "text": "", "image_term": ""})
        st.rerun()
    script["outro"] = st.text_area("Cierre", script.get("outro", ""), height=90)
    c1, c2 = st.columns([1, 3])
    if c1.button("💾 Guardar guion", type="primary"):
        try:
            studio.save_script(slug, script)
            st.success("Guion guardado.")
        except ValueError as exc:
            st.error(f"Revisa el guion: {exc}")
    c2.download_button("Descargar JSON", json.dumps(script, ensure_ascii=False, indent=2), file_name=f"{slug}.json")


@st.fragment(run_every=2)
def script_job(slug: str, job_id: str) -> None:
    info = studio.job_info(job_id)
    if info["state"] == "running":
        st.progress(info["progress"], text=info["stage"])
    elif info["state"] == "done":
        st.success("Guion escrito. Revísalo abajo y guarda los cambios.")
        st.session_state.pop(f"script-job::{slug}", None)
        st.session_state.pop(f"script::{slug}", None)
        st.rerun()
    else:
        st.error("No se pudo escribir el guion.")
        for line in info["errors"]:
            st.caption(line.split(" - ", 1)[-1])


def _voice_picker(label: str, current: str, key: str) -> str:
    groups = studio.voice_groups()
    names = [g for g in groups if groups[g]] + ["Otra (escribe el id)"]
    group_now = next((g for g in names[:-1] if current in groups[g]), names[-1])
    group = st.selectbox(f"{label}: proveedor", names, index=names.index(group_now), key=f"{key}-group")
    if group == names[-1]:
        return st.text_input("Id de la voz", current, key=f"{key}-custom", help="Ej.: elevenlabs:SzWoawNFBEew3QRjsGA6 o gcloud:es-US-Neural2-B")
    options = groups[group]
    return st.selectbox(label, options, index=options.index(current) if current in options else 0, key=f"{key}-voice")


def page_voice() -> None:
    from app.services import voice_google

    slug = require_project()
    settings = settings_for(slug)
    st.title("Voz")
    left, right = st.columns([3, 2], gap="large")
    with left:
        settings.voice_name = _voice_picker("Voz principal", settings.voice_name, "main")
        c1, c2 = st.columns(2)
        settings.voice_rate = c1.slider("Velocidad", 0.7, 1.4, float(settings.voice_rate), 0.05)
        settings.voice_volume = c2.slider("Volumen", 0.5, 1.5, float(settings.voice_volume), 0.05)
        c1, c2 = st.columns(2)
        settings.pause = c1.slider("Pausa entre frases (s)", 0.0, 1.2, float(settings.pause), 0.05,
                                   help="Respiración entre frases para una narración calmada (voces Gemini, Cloud TTS y ElevenLabs)")
        settings.voice_polish = c2.toggle("Masterizar la voz (sonido de estudio)", settings.voice_polish,
                                          help="Ecualización, de-esser, compresión suave y volumen de YouTube en dos pasadas")
        presets = list(studio.STYLE_PRESETS) + ["personalizado"]
        style_now = settings.voice_style if settings.voice_style in studio.STYLE_PRESETS else "personalizado"
        style = st.radio("Estilo (voces Gemini)", presets, index=presets.index(style_now), horizontal=True)
        if style == "personalizado":
            settings.voice_style = st.text_area("Indicaciones de tono", settings.voice_style if style_now == "personalizado" else "",
                                                placeholder="Habla como un divulgador joven y curioso...")
        else:
            settings.voice_style = style
            st.caption(voice_google.VOICE_STYLE_PRESETS[style])
        if st.button("💾 Guardar voz", type="primary"):
            save_settings(slug, settings)
            st.success("Voz guardada en el proyecto.")
    with right:
        st.markdown("### Probar")
        text = st.text_area("Texto de prueba", voice_google.LAB_TEXT, height=140)
        if st.button("▶️ Escuchar esta voz"):
            with st.spinner("Generando audio..."):
                results = voice_google.audition([settings.voice_name], text, str(studio.studio_dir("voice-test", slug)),
                                                rate=settings.voice_rate, style=settings.voice_style)
            if results[0]["ok"]:
                st.audio(results[0]["file"])
            else:
                st.error(results[0]["error"])
        st.markdown("### Precio aproximado (video de 10 min)")
        st.table({
            "Proveedor": ["Gemini Flash TTS", "Gemini Pro TTS", "Cloud TTS Chirp 3 HD", "ElevenLabs", "Edge"],
            "US$ aprox.": [f"${voice_google.estimate_cost(9000, p):.2f}" for p in ("gemini-flash", "gemini-pro", "gcloud-chirp", "elevenlabs", "edge")],
        })
        st.caption("Precios de lista 2025, revisa los actuales. Los créditos de ElevenLabs se renuevan cada mes.")
    st.divider()
    st.markdown("### Laboratorio de voces")
    st.caption("Compara varias voces con el mismo texto y el mismo estilo.")
    everything = sorted({v for group in studio.voice_groups().values() for v in group})
    chosen = st.multiselect("Voces", everything, default=list(voice_google.SPANISH_MALE_SHORTLIST[:4]), max_selections=8)
    if st.button("🎧 Comparar") and chosen:
        with st.spinner("Narrando con cada voz..."):
            st.session_state["lab"] = voice_google.audition(chosen, text, str(studio.studio_dir("voice-lab", slug)),
                                                            rate=settings.voice_rate, style=settings.voice_style)
    for result in st.session_state.get("lab", []):
        c1, c2, c3 = st.columns([2, 3, 1])
        c1.markdown(f"**{result['voice']}**")
        if result["ok"]:
            c2.audio(result["file"])
            if c3.button("Usar", key=f"use-{result['voice']}"):
                settings.voice_name = result["voice"]
                save_settings(slug, settings)
                st.rerun()
        else:
            c2.error(result["error"])


def page_style() -> None:
    slug = require_project()
    s = settings_for(slug)
    st.title("Estilo y edición")
    s.look = st.radio(
        "Estilo visual", studio.LOOKS, index=studio.LOOKS.index(s.look) if s.look in studio.LOOKS else 0, horizontal=True,
        format_func=lambda v: {"doodle": "✏️ Dibujado sobre color (estilo Cápsula)", "footage": "🎞️ Videos de stock + imágenes y escenas"}[v],
    )
    if s.look == "doodle":
        c1, c2, c3, c4 = st.columns(4)
        s.canvas_color = c1.color_picker("Color del fondo", s.canvas_color or "#F4C24F")
        s.boil = c2.toggle("Trazo vivo (hecho a mano)", s.boil, help="Los dibujos tiemblan muy levemente, como animación hecha a mano")
        s.max_drawings = c3.number_input("Máximo de dibujos IA", 0, 600, int(s.max_drawings), step=10,
                                         help="Cada dibujo cuesta ~US$0,02-0,04 y cada animación usa 2-4. Un video de 5 min "
                                              "usa unos 200-260 (~US$5-10). Después se usan fotos reales")
        logos = ["ninguno", "nutria"]
        s.logo = "nutria" if c4.selectbox("Logo en la esquina", logos, index=1 if s.logo == "nutria" else 0) == "nutria" else ""
        c1, c2, c3 = st.columns(3)
        s.shot_seconds = c1.slider("Cambio de imagen cada (s)", 2.0, 6.0, float(s.shot_seconds), 0.5,
                                   help="Como un animatic: escenas cortas y continuas que cuentan la historia")
        s.clips = c2.selectbox("Videos reales dentro del dibujo", studio.CLIPS, index=studio.CLIPS.index(s.clips) if s.clips in studio.CLIPS else 1,
                               format_func=lambda v: {"none": "Ninguno", "some": "Algunos (~8 %)", "more": "Más (~15 %)"}[v],
                               help="Clips de Pexels/Pixabay en un marco sobre el fondo, para lugares, naturaleza y máquinas reales")
        s.drawing_style = c3.selectbox("Estilo de dibujo", studio.DRAWING_STYLES,
                                       index=studio.DRAWING_STYLES.index(s.drawing_style) if s.drawing_style in studio.DRAWING_STYLES else 0,
                                       format_func=lambda v: {"cartoon": "Caricatura animada (pulida)", "ink": "Tinta a mano (garabato)"}[v])
        c1, c2 = st.columns(2)
        s.memes = c1.selectbox("Reacciones tipo meme", studio.MEMES, index=studio.MEMES.index(s.memes) if s.memes in studio.MEMES else 0,
                               format_func=lambda v: {"off": "No", "otter": "La nutria reacciona (dibujada, sin derechos de autor)",
                                                      "folder": "Mi carpeta de memes (por emoción)"}[v],
                               help="Cortes de ~2 s en los remates o datos sorprendentes, como mucho uno cada 40 s")
        if s.memes == "folder":
            s.memes_dir = c2.text_input("Carpeta de memes", s.memes_dir, placeholder="resource/memes",
                                        help="Subcarpetas por emoción: sorpresa, risa, facepalm, mente, confundido, miedo, triste...")
        st.caption("Casi todo son dibujos de IA a pantalla completa: animaciones cortas (un dibujo por segundo que "
                   "continúa del anterior) e ilustraciones con un movimiento de cámara suave. De vez en cuando, una escena "
                   "explicativa con imágenes o un video real enmarcado.")
    look, motion, sound, languages = st.tabs(["🎨 Imagen", "🎬 Edición", "🔊 Sonido", "🌎 Idiomas"])
    with look:
        c1, c2, c3 = st.columns(3)
        s.aspect = c1.selectbox("Formato", studio.ASPECTS, index=studio.ASPECTS.index(s.aspect), help="16:9 YouTube, 9:16 Shorts")
        s.accent = c2.color_picker("Color de la marca", s.accent or "#FF4F5E")
        use_scene_color = c3.toggle("Color propio para las escenas", bool(s.scene_color))
        s.scene_color = c3.color_picker("Color de las escenas", s.scene_color or "#FFE3E6") if use_scene_color else ""
        c1, c2 = st.columns(2)
        assets = c1.selectbox("Presentador", ["nutria", "demo", "ninguno", "carpeta propia"],
                              index=["nutria", "demo", ""].index(s.assets) if s.assets in ("nutria", "demo", "") else 3)
        if assets == "carpeta propia":
            s.assets = c1.text_input("Carpeta del personaje", s.assets if s.assets not in ("nutria", "demo") else "")
        else:
            s.assets = "" if assets == "ninguno" else assets
        s.illustrations = c2.radio("Dibujos en escenas", studio.ILLUSTRATIONS, index=studio.ILLUSTRATIONS.index(s.illustrations),
                                   format_func=lambda v: {"icons": "Iconos OpenMoji (gratis)", "ai": "Ilustraciones Imagen (~US$0,02 c/u)"}[v])
        s.beats = c2.radio("Imágenes flotantes", studio.BEAT_MODES, index=studio.BEAT_MODES.index(s.beats),
                           format_func=lambda v: {"web": "Fotos y diagramas reales", "ai": "Dibujadas por IA", "none": "Ninguna"}[v])
        s.picture_check = st.toggle("Gemini revisa cada imagen (recomendado)", s.picture_check)
        s.subtitles = st.toggle("Subtítulos quemados", s.subtitles)
    with motion:
        c1, c2, c3 = st.columns(3)
        s.host = c1.selectbox("Nutria en pantalla", studio.HOST_MODES, index=studio.HOST_MODES.index(s.host),
                              format_func=lambda v: {"auto": "Va y viene", "always": "Todo el video", "none": "Nunca"}[v])
        s.host_presence = c2.select_slider("Cuánto aparece", studio.PRESENCES, value=s.host_presence,
                                           format_func=lambda v: {"low": "Poco", "normal": "Normal", "high": "Mucho"}[v])
        s.lip_sync = c3.toggle("Boca que habla", s.lip_sync)
        c1, c2, c3 = st.columns(3)
        s.scenes = c1.toggle("Escenas explicativas", s.scenes)
        s.openers = c2.toggle("Portada de cada sección", s.openers, help="Número, título e imagen estrictamente del tema")
        s.progress_bar = c3.toggle("Barra de progreso", s.progress_bar)
        c1, c2, c3 = st.columns(3)
        s.subscribe = c1.selectbox("Animación de suscribirse", studio.SUBSCRIBE_MODES, index=studio.SUBSCRIBE_MODES.index(s.subscribe),
                                   format_func=lambda v: {"both": "Inicio y final", "intro": "Inicio", "outro": "Final", "none": "Nunca"}[v])
        s.numbers = c2.toggle("Numerar secciones", s.numbers)
        s.item_titles = c3.toggle("Títulos de sección", s.item_titles)
        c1, c2 = st.columns(2)
        s.gap = c1.slider("Pausa entre secciones (s)", 0.0, 1.5, float(s.gap), 0.1)
        s.zoom = c2.slider("Zoom lento de fondos", 0.0, 0.2, float(s.zoom), 0.01)
    with sound:
        s.sound_effects = st.toggle("Efectos de sonido", s.sound_effects)
        s.sfx_volume = st.slider("Volumen de efectos", 0.0, 1.5, float(s.sfx_volume), 0.05, disabled=not s.sound_effects)
        s.music = st.toggle("Música de fondo (storage/bgm)", s.music)
        s.music_volume = st.slider("Volumen de música", 0.0, 1.0, float(s.music_volume), 0.05, disabled=not s.music)
    with languages:
        st.caption("Se hace una versión por idioma con su propio plan de edición, reutilizando las mismas imágenes.")
        extra = st.multiselect("También en", [lang for lang in LANGUAGES if lang != s.language], default=list(s.also))
        s.also = {language: _voice_picker(f"Voz {language}", s.also.get(language, ""), f"also-{language}") for language in extra}
    if st.button("💾 Guardar estilo", type="primary"):
        save_settings(slug, s)
        st.success("Estilo guardado.")


def page_render() -> None:
    slug = require_project()
    s = settings_for(slug)
    st.title("Producir")
    script = studio.load_script(slug)
    if script is None:
        st.warning("Este proyecto aún no tiene guion guardado.")
        st.stop()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Secciones", len(script["items"]))
    c2.metric("Duración", f"~{studio.estimated_minutes(script)} min")
    c3.metric("Voz", s.voice_name.split(":")[-1][:18])
    c4.metric("Formato", s.aspect)
    plan = studio.project_dir(slug) / "edit-plan.json"
    use_plan = plan.is_file() and st.toggle("Usar el plan de edición guardado en el proyecto", True)
    argv = studio.build_argv(s, script_file=str(studio.script_path(slug)), edit_plan=str(plan) if use_plan else "")
    with st.expander("Comando equivalente"):
        st.code("python list_video.py " + " ".join(f'"{a}"' if " " in a else a for a in argv), language="bash")
    if st.button("🎬 Renderizar video", type="primary"):
        job = studio.start_job(argv, label=script.get("title", slug), project=slug)
        st.toast(f"Render iniciado: {job}")
    st.divider()
    jobs_panel(slug)


@st.fragment(run_every=3)
def jobs_panel(slug: str) -> None:
    jobs = [j for j in studio.list_jobs(limit=12, project=slug) if j["tool"] == "list_video.py"]
    if not jobs:
        st.caption("Todavía no hay renders.")
    for job in jobs:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{job['label'] or job['id']}** · {STATE_LABELS.get(job['state'], job['state'])} · {when(job['started'])}")
            if job["state"] == "running":
                c1.progress(job["progress"], text=job["stage"])
                if c2.button("Cancelar", key=f"cancel-{job['id']}"):
                    studio.cancel_job(job["id"])
            elif job["state"] == "done" and job["summary"]:
                if c2.button("Ver resultado", key=f"see-{job['id']}"):
                    st.session_state["result-job"] = job["id"]
                    st.switch_page(PAGES["results"])
            for line in job["errors"]:
                st.error(line.split(" - ", 1)[-1])


def page_results() -> None:
    slug = require_project()
    st.title("Resultados")
    done = [j for j in studio.list_jobs(limit=30, project=slug) if j["state"] == "done" and j["summary"] and j["summary"].get("result")]
    if not done:
        st.info("Cuando termine un render aparecerá aquí.")
        st.stop()
    ids = [j["id"] for j in done]
    chosen = st.session_state.get("result-job") if st.session_state.get("result-job") in ids else ids[0]
    job_id = st.selectbox("Render", ids, index=ids.index(chosen), format_func=lambda i: f"{next(j['label'] for j in done if j['id'] == i)} · {i}")
    summary = next(j["summary"] for j in done if j["id"] == job_id)
    versions = [("principal", summary["task_id"])] + [(v["language"], v["task_id"]) for v in summary.get("also") or []]
    tabs = st.tabs([name for name, _ in versions])
    script = studio.load_script(slug) or {}
    for tab, (name, task_id) in zip(tabs, versions):
        with tab:
            outputs = studio.task_outputs(task_id)
            left, right = st.columns([3, 2], gap="large")
            with left:
                for video in outputs["videos"]:
                    st.video(video)
                    with open(video, "rb") as fp:
                        st.download_button("⬇️ Descargar MP4", fp, file_name=f"{slug}-{name}.mp4", key=f"dl-{task_id}")
                st.caption(f"Carpeta: `{outputs['folder']}`")
                if outputs.get("report"):
                    with st.expander("Informe del render (dibujos, respaldos y advertencias)"):
                        st.code(outputs["report"], language=None)
            with right:
                st.markdown("### Descripción para YouTube")
                hashtags = st.text_input("Hashtags", "#ciencia #educación #cabeceando", key=f"tags-{task_id}")
                description = studio.youtube_description(
                    script.get("title", ""), script.get("intro", ""), outputs["chapters"], outputs["credits"], hashtags.split()
                )
                st.text_area("Copia y pega", description, height=380, key=f"desc-{task_id}")
                st.download_button("Descargar descripción", description, file_name=f"{slug}-{name}-descripcion.txt", key=f"ddl-{task_id}")


def page_plan() -> None:
    from app.services import llm

    slug = require_project()
    st.title("Plan de edición")
    st.caption("Lo que la IA decidió mostrar en cada momento. Edítalo y vuelve a renderizar con **Producir**.")
    project_plan = studio.project_dir(slug) / "edit-plan.json"
    sources = {}
    if project_plan.is_file():
        sources["Plan del proyecto"] = str(project_plan)
    for job in studio.list_jobs(limit=20, project=slug):
        summary = job.get("summary") or {}
        if summary.get("task_id"):
            plan = studio.task_outputs(summary["task_id"])["plan"]
            if plan:
                sources[f"Render {job['id']}"] = plan
    if not sources:
        st.info("Renderiza una vez para obtener el plan.")
        st.stop()
    source = st.selectbox("Plan", list(sources))
    data = json.loads(Path(sources[source]).read_text(encoding="utf-8"))
    segments = data.get("segments") or []
    script = studio.load_script(slug) or {}
    names = ["Intro"] + [i.get("name", "") for i in script.get("items") or []] + ["Cierre"]
    for number, entry in enumerate(segments):
        scenes = ", ".join(sc.get("type", "") for sc in entry.get("scenes") or []) or "—"
        beats = len([b for b in entry.get("beats") or [] if b.get("type") != "react"])
        st.markdown(f"**{names[number] if number < len(names) else number}** · escenas: {scenes} · imágenes: {beats}")
    edited = st.text_area("JSON", json.dumps(data, ensure_ascii=False, indent=2), height=420)
    if st.button("💾 Guardar como plan del proyecto", type="primary"):
        try:
            parsed = json.loads(edited)
            expressions = sorted({e.get("expression") for e in parsed.get("segments") or [] if e.get("expression")})
            clean = llm.normalize_edit_plan(parsed, len(parsed.get("segments") or []), expressions)
            project_plan.write_text(json.dumps({"segments": clean}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            st.success("Plan guardado. Actívalo en **Producir**.")
        except Exception as exc:
            st.error(f"El JSON no es válido: {exc}")


def page_research() -> None:
    st.title("Investigación de canales")
    st.caption("Estadísticas públicas de canales parecidos, los videos que más destacan y (opcional) ideas de la IA.")
    with st.form("research"):
        channels = st.text_area("Canales (uno por línea)", placeholder="@CapsulaMental\n@QuantumFracture")
        searches = st.text_input("Buscar también", placeholder="explicado para principiantes")
        analyze = st.toggle("Pedir patrones e ideas al LLM", True)
        brief = st.text_input("Tu canal", "Cabeceando: temas generales explicados para nutrias, videos largos en formato lista")
        if st.form_submit_button("🔎 Investigar", type="primary"):
            argv = [a for c in channels.splitlines() if c.strip() for a in ("--channel", c.strip())]
            argv += [a for q in searches.split(",") if q.strip() for a in ("--search", q.strip())]
            if analyze:
                argv += ["--analyze", "--brief", brief]
            if argv:
                st.session_state["research-job"] = studio.start_job(argv, label="Investigación", tool="research.py")
    job_id = st.session_state.get("research-job")
    if job_id:
        research_job(job_id)


@st.fragment(run_every=3)
def research_job(job_id: str) -> None:
    info = studio.job_info(job_id)
    if info["state"] == "running":
        st.info("Investigando... esto puede tardar un par de minutos.")
        return
    summary = info["summary"] or {}
    if info["state"] != "done":
        st.error("La investigación falló.")
        for line in info["errors"]:
            st.caption(line.split(" - ", 1)[-1])
        return
    analysis = summary.get("analysis")
    if analysis and os.path.isfile(analysis):
        st.markdown(Path(analysis).read_text(encoding="utf-8"))
    for key in ("csv", "output"):
        if summary.get(key) and os.path.isfile(summary[key]):
            with open(summary[key], "rb") as fp:
                st.download_button("Descargar CSV", fp, file_name=os.path.basename(summary[key]))


def page_settings() -> None:
    st.title("Ajustes")
    values = studio.read_settings()
    fields = {key: (section, key, label, kind, help_text) for section, key, label, kind, help_text in studio.SETTINGS_FIELDS}
    with st.form("settings"):
        columns = st.columns(len(studio.SETTINGS_GROUPS), gap="large")
        for column, (title, keys) in zip(columns, studio.SETTINGS_GROUPS):
            with column:
                st.markdown(f"### {title}")
                for key in keys:
                    section, key, label, kind, help_text = fields[key]
                    name = f"{section}.{key}"
                    if kind == "bool":
                        values[name] = st.toggle(label, bool(values[name]), help=help_text)
                    elif kind == "float":
                        values[name] = st.number_input(label, value=float(values[name] or 0), step=0.5, help=help_text)
                    elif kind == "provider":
                        providers = studio.llm_providers()
                        current = str(values[name] or "gemini")
                        values[name] = st.selectbox(label, providers, index=providers.index(current) if current in providers else 0, help=help_text)
                    else:
                        values[name] = st.text_input(label, str(values[name] or ""), type="password" if kind == "secret" else "default", help=help_text)
        if st.form_submit_button("💾 Guardar en config.toml", type="primary"):
            studio.write_settings(values)
            st.success("Ajustes guardados.")
    st.markdown("### Comprobación")
    for name, ok, detail in studio.diagnostics():
        st.markdown(f"{'✅' if ok else '⚠️'} **{name}** <span class='cab-muted'>{detail}</span>", unsafe_allow_html=True)
    with st.expander("Cómo conectar Google (Vertex AI y voces) en Fedora o CachyOS"):
        st.code(
            "# Fedora\nsudo dnf install google-cloud-cli ffmpeg\n"
            "# CachyOS / Arch\nsudo pacman -S ffmpeg && yay -S google-cloud-cli\n\n"
            "gcloud auth application-default login\n"
            "gcloud auth application-default set-quota-project TU_PROYECTO\n"
            "gcloud services enable aiplatform.googleapis.com texttospeech.googleapis.com",
            language="bash",
        )


PAGES = {
    "projects": st.Page(page_projects, title="Proyectos", icon="🗂️", default=True),
    "script": st.Page(page_script, title="Guion", icon="✍️", url_path="guion"),
    "voice": st.Page(page_voice, title="Voz", icon="🎙️", url_path="voz"),
    "style": st.Page(page_style, title="Estilo", icon="🎨", url_path="estilo"),
    "render": st.Page(page_render, title="Producir", icon="🎬", url_path="producir"),
    "results": st.Page(page_results, title="Resultados", icon="📦", url_path="resultados"),
    "plan": st.Page(page_plan, title="Plan de edición", icon="🧩", url_path="plan"),
    "research": st.Page(page_research, title="Investigación", icon="🔎", url_path="investigacion"),
    "settings": st.Page(page_settings, title="Ajustes", icon="⚙️", url_path="ajustes"),
}


def main() -> None:
    navigation = st.navigation({
        "Crear": [PAGES["projects"], PAGES["script"], PAGES["voice"], PAGES["style"]],
        "Producir": [PAGES["render"], PAGES["results"], PAGES["plan"]],
        "Más": [PAGES["research"], PAGES["settings"]],
    })
    sidebar()
    navigation.run()


if __name__ == "__main__":  # streamlit run studio/Studio.py
    main()
