"""
Cabeceando Studio: everything the desktop GUI (studio/Studio.py) does that is
not drawing widgets, so it can be tested.

* Projects live in storage/studio/projects/<slug>/ with the script
  (script.json), the render settings (settings.json) and the renders made.
* Renders run list_video.py in a background process (a "job"), so a long
  render survives the browser tab; the GUI follows its log and progress.
* Helpers build the YouTube description, list the voices, edit config.toml
  and check that everything a render needs is in place.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
ASPECTS = ("16:9", "9:16", "1:1")
HOST_MODES = ("auto", "always", "none")
PRESENCES = ("low", "normal", "high")
BEAT_MODES = ("web", "ai", "none")
SUBSCRIBE_MODES = ("both", "intro", "outro", "none")
ILLUSTRATIONS = ("icons", "ai")
STYLE_PRESETS = ("divulgador", "entusiasta", "profe", "calmado", "sereno", "narrador")
LOOKS = ("doodle", "footage")
FORMATS = ("story", "list")
SETTINGS_VERSION = 3
CLIPS = ("none", "some", "more")
MEMES = ("off", "otter", "folder")
DRAWING_STYLES = ("cartoon", "flat", "ink")
IMAGE_QUALITIES = ("economy", "standard", "high", "max")
REVIEW_ACTIONS = ("keep", "redo", "photo", "remove")


def studio_dir(*parts: str) -> Path:
    from app.utils import utils

    path = Path(utils.storage_dir("studio", create=True)).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Render settings and the command line they become
# ---------------------------------------------------------------------------


@dataclass
class RenderSettings:
    # Content
    language: str = "es-CO"
    script_format: str = "story"  # story (one continuous narrative) or list (numbered sections)
    items: int = 6
    words_per_item: int = 130
    # Voice
    voice_name: str = "gemini:Schedar-Even"
    voice_rate: float = 1.0
    voice_volume: float = 1.0
    voice_style: str = "calmado"
    pause: float = 0.5  # breath between sentences (Gemini, Cloud TTS, ElevenLabs)
    voice_polish: bool = True  # studio mastering of the voice
    # Look
    look: str = "doodle"  # doodle (everything drawn) or footage (stock video + scenes)
    canvas_color: str = "#F4C24F"
    boil: bool = True
    logo: str = ""  # "nutria" adds the round channel badge in the corner
    max_drawings: int = 260
    shot_seconds: float = 5.0  # a new picture about this often (a calm documentary pace)
    image_quality: str = "standard"  # AI drawings: economy, standard, high or max
    ai_videos: int = 0  # illustrations brought to life by Veo (paid per second)
    director_review: bool = True  # a film editor pass corrects the storyboard before drawing
    clips: str = "some"  # real video clips in a frame now and then
    memes: str = "off"  # comic reactions: off, otter or folder
    memes_dir: str = ""
    drawing_style: str = "cartoon"
    aspect: str = "16:9"
    assets: str = "nutria"
    accent: str = "#FF4F5E"
    scene_color: str = ""
    host: str = "auto"
    host_presence: str = "low"
    lip_sync: bool = False
    scenes: bool = True
    openers: bool = True
    illustrations: str = "icons"
    beats: str = "web"
    picture_check: bool = True
    progress_bar: bool = False
    subscribe: str = "both"
    numbers: bool = True
    item_titles: bool = True
    # Sound
    sound_effects: bool = True
    sfx_volume: float = 0.65
    music: bool = False
    music_volume: float = 0.2
    # Subtitles
    subtitles: bool = False
    # Pace
    gap: float = 0.4
    zoom: float = 0.08
    # Other languages: {"en-US": "gemini:Puck-Upbeat"}
    also: Dict[str, str] = field(default_factory=dict)
    # Settings saved before a version get its new defaults (see from_dict).
    version: int = SETTINGS_VERSION

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "RenderSettings":
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in (data or {}).items() if k in known}
        if values and int(values.get("version") or 1) < 2:
            # Round 7: no corner badge for now, and room for the many more drawings of the animated look.
            values["logo"] = ""
            if int(values.get("max_drawings") or 0) == 160:
                values["max_drawings"] = 260
        if values and int(values.get("version") or 1) < 3 and float(values.get("shot_seconds") or 3.0) <= 3.0:
            # Round 8: a calmer, documentary pace.
            values["shot_seconds"] = 5.0
        values["version"] = SETTINGS_VERSION
        settings = cls(**values)
        settings.also = {str(k): str(v) for k, v in dict(settings.also or {}).items() if str(k).strip()}
        return settings

    def to_dict(self) -> dict:
        return asdict(self)


def _number(value: float) -> str:
    return f"{float(value):g}"


def build_argv(
    settings: RenderSettings,
    *,
    script_file: str = "",
    subject: str = "",
    script_only: bool = False,
    edit_plan: str = "",
    task_id: str = "",
    output: str = "",
    review: bool = False,
) -> List[str]:
    """The list_video.py arguments for these settings (the GUI and the CLI stay in step).

    ``review`` makes every picture and stops before rendering (see ``load_review``).
    """
    if bool(script_file) == bool(subject):
        raise ValueError("give either a script file or a subject")
    s = settings
    argv = ["--script", script_file] if script_file else ["--subject", subject]
    argv += ["--format", s.script_format if s.script_format in FORMATS else "list"]
    if subject:
        argv += ["--items", str(int(s.items))]
        if s.words_per_item:
            argv += ["--words-per-item", str(int(s.words_per_item))]
        if script_only:
            argv.append("--script-only")
        if output:
            argv += ["--output", output]
    if task_id:
        argv += ["--task-id", task_id]
    argv += ["--video-language", s.language, "--video-aspect", s.aspect]
    argv += ["--voice-name", s.voice_name, "--voice-rate", _number(s.voice_rate), "--voice-volume", _number(s.voice_volume)]
    if s.voice_style:
        argv += ["--voice-style", s.voice_style]
    argv += ["--pause", _number(s.pause)]
    if not s.voice_polish:
        argv.append("--no-voice-polish")
    argv += ["--look", s.look if s.look in LOOKS else "footage"]
    if s.look == "doodle":
        if s.canvas_color:
            argv += ["--canvas-color", s.canvas_color]
        if not s.boil:
            argv.append("--no-boil")
        argv += ["--max-drawings", str(int(s.max_drawings))]
        argv += ["--shot-seconds", _number(s.shot_seconds), "--clips", s.clips if s.clips in CLIPS else "some"]
        argv += ["--drawing-style", s.drawing_style if s.drawing_style in DRAWING_STYLES else "cartoon"]
        argv += ["--image-quality", s.image_quality if s.image_quality in IMAGE_QUALITIES else "standard"]
        if int(s.ai_videos) > 0:
            argv += ["--ai-videos", str(int(s.ai_videos))]
        if not s.director_review:
            argv.append("--no-director-review")
        if review:
            argv.append("--review")
    if s.memes in MEMES and s.memes != "off":
        argv += ["--memes", s.memes]
        if s.memes == "folder" and s.memes_dir:
            argv += ["--memes-dir", s.memes_dir]
    if s.logo:
        argv += ["--logo", s.logo]
    argv += ["--gap", _number(s.gap), "--zoom", _number(s.zoom)]
    if not s.numbers:
        argv.append("--no-numbers")
    if not s.item_titles:
        argv.append("--no-item-titles")
    if s.assets:
        argv += ["--assets", s.assets]
    argv += ["--host", s.host, "--host-presence", s.host_presence]
    if s.lip_sync:
        argv.append("--lip-sync")
    if not s.scenes:
        argv.append("--no-scenes")
    if not s.openers:
        argv.append("--no-openers")
    argv += ["--illustrations", s.illustrations, "--beats", s.beats, "--subscribe", s.subscribe]
    if not s.picture_check:
        argv.append("--no-picture-check")
    if s.progress_bar:
        argv.append("--progress-bar")
    if s.accent:
        argv += ["--accent", s.accent]
    if s.scene_color:
        argv += ["--scene-color", s.scene_color]
    if s.sound_effects:
        argv += ["--sfx-volume", _number(s.sfx_volume)]
    else:
        argv.append("--no-sfx")
    if s.music:
        argv += ["--bgm-type", "random", "--bgm-volume", _number(s.music_volume)]
    argv.append("--subtitle-enabled" if s.subtitles else "--no-subtitle-enabled")
    if edit_plan:
        argv += ["--edit-plan", edit_plan]
    if s.also:
        argv += ["--also-in", ",".join(s.also)]
        for language, voice_name in s.also.items():
            if voice_name:
                argv += ["--also-voice", f"{language}={voice_name}"]
    return argv


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "proyecto"


def project_dir(slug: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]+", slug or ""):
        raise ValueError(f"invalid project name: {slug!r}")
    return studio_dir("projects") / slug


def create_project(title: str) -> str:
    base = slugify(title)
    slug, number = base, 2
    while (studio_dir("projects") / slug).exists():
        slug, number = f"{base}-{number}", number + 1
    folder = project_dir(slug)
    _write_json(folder / "project.json", {"title": title.strip() or slug, "created": time.time(), "renders": []})
    _write_json(folder / "settings.json", RenderSettings().to_dict())
    return slug


def list_projects() -> List[dict]:
    projects = []
    for folder in studio_dir("projects").iterdir():
        meta = _read_json(folder / "project.json")
        if not folder.is_dir() or meta is None:
            continue
        updated = max((p.stat().st_mtime for p in folder.iterdir()), default=folder.stat().st_mtime)
        projects.append({"slug": folder.name, "title": meta.get("title") or folder.name, "updated": updated,
                         "renders": len(meta.get("renders") or []), "has_script": (folder / "script.json").is_file()})
    return sorted(projects, key=lambda p: p["updated"], reverse=True)


def project_meta(slug: str) -> dict:
    return _read_json(project_dir(slug) / "project.json") or {"title": slug, "renders": []}


def script_path(slug: str) -> Path:
    return project_dir(slug) / "script.json"


def load_script(slug: str) -> Optional[dict]:
    return _read_json(script_path(slug))


def validate_script(data: dict) -> Tuple[Optional[dict], str]:
    """(clean script, "") or (None, what is wrong)."""
    from app.models.schema import ListVideoScript

    if not isinstance(data, dict):
        return None, "el archivo debe ser un objeto JSON con title, intro, items y outro"
    data = dict(data)
    items = data.get("items")
    if isinstance(items, list):
        # A section left completely blank (the editor's placeholder) is skipped.
        data["items"] = [
            item for item in items
            if not isinstance(item, dict) or any(str(v or "").strip() for v in item.values())
        ]
    problems = []
    if not str(data.get("title") or "").strip():
        problems.append('falta el título ("title")')
    if not isinstance(data.get("items"), list) or not data["items"]:
        problems.append('no hay secciones ("items")')
    else:
        for number, item in enumerate(data["items"], 1):
            if not isinstance(item, dict):
                problems.append(f"la sección {number} no es un objeto")
                continue
            missing = [label for key, label in (("name", 'nombre ("name")'), ("text", 'narración ("text")'))
                       if not str(item.get(key) or "").strip()]
            if missing:
                problems.append(f"la sección {number} no tiene {' ni '.join(missing)}")
    if problems:
        return None, "; ".join(problems)
    try:
        return ListVideoScript.model_validate(data).model_dump(), ""
    except Exception as exc:  # pydantic.ValidationError: unknown keys, texts too long...
        return None, str(exc)


def save_script(slug: str, data: dict) -> Path:
    clean, error = validate_script(data)
    if clean is None:
        raise ValueError(error)
    path = script_path(slug)
    _write_json(path, clean)
    return path


def load_settings(slug: str) -> RenderSettings:
    return RenderSettings.from_dict(_read_json(project_dir(slug) / "settings.json"))


def save_settings(slug: str, settings: RenderSettings) -> None:
    _write_json(project_dir(slug) / "settings.json", settings.to_dict())


def record_render(slug: str, job_id: str, kind: str = "render") -> None:
    meta = project_meta(slug)
    meta.setdefault("renders", []).append({"job": job_id, "kind": kind, "date": time.time()})
    _write_json(project_dir(slug) / "project.json", meta)


def word_count(script: dict) -> int:
    texts = [script.get("intro", ""), script.get("outro", "")] + [i.get("text", "") for i in script.get("items") or []]
    return sum(len(re.findall(r"\w+", t or "")) for t in texts)


def estimated_minutes(script: dict) -> float:
    """About 150 spoken words per minute in Spanish, plus the openers."""
    return round(word_count(script) / 150.0 + 3.5 * len(script.get("items") or []) / 60.0, 1)


# ---------------------------------------------------------------------------
# Jobs: list_video.py in the background
# ---------------------------------------------------------------------------

_STAGES = (
    (re.compile(r"generating list script"), 0.03, "Escribiendo el guion con IA"),
    (re.compile(r"list script generated"), 0.08, "Guion listo"),
    (re.compile(r"list video narration \[(\d+)/(\d+)\]"), (0.08, 0.22), "Narrando la sección {0} de {1}"),
    (re.compile(r"visual bible written"), 0.29, "Dirección de arte lista (personajes y fotos de archivo)"),
    (re.compile(r"storyboard generated"), 0.31, "Guion gráfico listo"),
    (re.compile(r"edit plan generated"), 0.32, "Plan de edición listo"),
    (re.compile(r"pictures added where only footage"), 0.35, "Imágenes extra añadidas"),
    (re.compile(r"character sheets ready"), 0.34, "Fichas de personajes dibujadas"),
    (re.compile(r"list video segment \[(\d+)/(\d+)\]"), (0.38, 0.57), "Montando la sección {0} de {1}"),
    (re.compile(r"rendering the (\S+) version"), 0.0, "Versión en {0}"),
    (re.compile(r"review ready"), 1.0, "Revisión lista: abre Revisar"),
    (re.compile(r"list video finished"), 1.0, "Video terminado"),
)
# Every picture made before the first section is assembled (drawings, real archive pictures, character sheets).
_PICTURE_LINE = re.compile(r" - (?:drawn|illustration drawn|character sheet drawn|archive picture|video made with \S+): ")


def progress_from_log(text: str) -> Tuple[float, str]:
    """(fraction done, what is happening) read from a render log."""
    fraction, stage = 0.0, "Preparando"
    pictures = 0
    for line in (text or "").splitlines():
        if _PICTURE_LINE.search(line):
            # Every picture is made while the first section is assembled: count them, so a long wait shows progress.
            pictures += 1
            if fraction <= 0.38:
                fraction, stage = max(fraction, 0.34), f"Preparando las imágenes: {pictures} listas"
            continue
        for pattern, value, label in _STAGES:
            match = pattern.search(line)
            if not match:
                continue
            groups = match.groups()
            if isinstance(value, tuple) and len(groups) == 2:
                done, total = int(groups[0]), max(1, int(groups[1]))
                fraction = value[0] + value[1] * (done - 1) / total
            elif label.startswith("Versión"):
                fraction = 0.0
            else:
                fraction = value
            stage = label.format(*groups)
    return min(1.0, fraction), stage


def parse_summary(text: str) -> Optional[dict]:
    """The JSON summary a tool prints last (list_video.py: task id, files, other languages)."""
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def errors_from_log(text: str, limit: int = 6) -> List[str]:
    lines = [line for line in (text or "").splitlines() if "| ERROR" in line or "| CRITICAL" in line or line.startswith("list_video.py: error")]
    return lines[-limit:]


TOOLS = ("list_video.py", "research.py", "voice_lab.py")


def start_job(argv: Sequence[str], label: str = "", project: str = "", tool: str = "list_video.py") -> str:
    """Run ``tool`` (list_video.py by default) with ``argv`` in the background; returns the job id."""
    if tool not in TOOLS:
        raise ValueError(f"unknown tool: {tool}")
    job_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid() % 1000:03d}{int(time.time() * 1000) % 1000:03d}"
    folder = studio_dir("jobs", job_id)
    _write_json(folder / "job.json", {"argv": list(argv), "label": label, "project": project, "tool": tool, "started": time.time()})
    _write_json(folder / "status.json", {"state": "running"})
    log = open(folder / "log.txt", "w", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", NO_COLOR="1")
    process = subprocess.Popen(
        [sys.executable, "-X", "utf8", "-m", "app.services.studio", "run", str(folder)],
        cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
    )
    log.close()
    _write_json(folder / "status.json", {"state": "running", "pid": process.pid})
    if project and tool == "list_video.py":
        record_render(project, job_id, "script" if "--script-only" in argv else "render")
    return job_id


def run_job(folder: str) -> int:
    """Inside the background process: run list_video.py and write how it ended."""
    folder_path = Path(folder)
    job = _read_json(folder_path / "job.json") or {}
    status = _read_json(folder_path / "status.json") or {}
    tool = job.get("tool", "list_video.py")
    if tool not in TOOLS:
        tool = "list_video.py"
    code = subprocess.call([sys.executable, "-X", "utf8", str(ROOT / tool), *job.get("argv", [])], cwd=str(ROOT))
    status.update(state="done" if code == 0 else "failed", exit_code=code, finished=time.time())
    _write_json(folder_path / "status.json", status)
    return code


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, TypeError):
        return False
    try:  # a finished child that was never reaped is still "alive" for kill(0)
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True
    except OSError:
        return False


def job_info(job_id: str) -> dict:
    if not re.fullmatch(r"[0-9-]+", job_id or ""):
        raise ValueError(f"invalid job id: {job_id!r}")
    folder = studio_dir("jobs", job_id)
    job = _read_json(folder / "job.json") or {}
    status = _read_json(folder / "status.json") or {}
    log = (folder / "log.txt").read_text(encoding="utf-8", errors="replace") if (folder / "log.txt").is_file() else ""
    log = _ANSI.sub("", log)
    state = status.get("state", "unknown")
    if state == "running" and status.get("pid") and not _alive(int(status["pid"])):
        state = "failed" if parse_summary(log) is None else "done"
    fraction, stage = progress_from_log(log)
    summary = parse_summary(log)
    if state == "done":
        fraction, stage = 1.0, "Terminado"
    return {
        "id": job_id, "label": job.get("label", ""), "project": job.get("project", ""), "argv": job.get("argv", []),
        "tool": job.get("tool", "list_video.py"),
        "started": job.get("started"), "finished": status.get("finished"), "state": state,
        "progress": fraction, "stage": stage, "summary": summary, "errors": errors_from_log(log),
        "log": log, "folder": str(folder),
    }


def list_jobs(limit: int = 30, project: str = "") -> List[dict]:
    folders = sorted((p for p in studio_dir("jobs").iterdir() if p.is_dir()), reverse=True)
    jobs = []
    for folder in folders:
        info = job_info(folder.name)
        if project and info["project"] != project:
            continue
        info.pop("log")
        jobs.append(info)
        if len(jobs) >= limit:
            break
    return jobs


def cancel_job(job_id: str) -> bool:
    status_file = studio_dir("jobs", job_id) / "status.json"
    status = _read_json(status_file) or {}
    pid = status.get("pid")
    if not pid or status.get("state") != "running":
        return False
    try:
        os.killpg(int(pid), signal.SIGTERM)
    except OSError:
        return False
    status.update(state="cancelled", finished=time.time())
    _write_json(status_file, status)
    return True


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def task_outputs(task_id: str) -> dict:
    from app.utils import utils

    if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id or ""):
        raise ValueError(f"invalid task id: {task_id!r}")
    folder = Path(utils.storage_dir()) / "tasks" / task_id

    def text(name: str) -> str:
        path = folder / name
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""

    return {
        "folder": str(folder),
        "videos": [str(p) for p in sorted(folder.glob("final-*.mp4"))],
        "chapters": text("chapters.txt"),
        "credits": text("credits.txt"),
        "report": text("render-report.txt"),
        "plan": str(folder / "edit-plan.json") if (folder / "edit-plan.json").is_file() else "",
        "script": str(folder / "list-script.json") if (folder / "list-script.json").is_file() else "",
    }


# ---------------------------------------------------------------------------
# Review before rendering
# ---------------------------------------------------------------------------


def new_task_id() -> str:
    """A task id chosen in advance, so a review and its final render share one folder (and its narration)."""
    from app.utils import utils

    return utils.get_uuid()


def load_review(task_id: str) -> Optional[dict]:
    """The review.json of a render prepared with --review (the shots, their pictures and what is said)."""
    folder = Path(task_outputs(task_id)["folder"])
    review = _read_json(folder / "review.json")
    if review is None or not isinstance(review.get("shots"), list):
        return None
    review["task_id"] = task_id
    return review


def review_jobs(slug: str) -> List[dict]:
    """Finished review preparations of a project, newest first: {"job", "task_id", "label", "started"}."""
    found = []
    for job in list_jobs(limit=40, project=slug):
        summary = job.get("summary") or {}
        if job["state"] == "done" and summary.get("review_file") and summary.get("task_id"):
            found.append({"job": job["id"], "task_id": summary["task_id"], "label": job["label"], "started": job["started"]})
    return found


_TAKE = re.compile(r" \(new take (\d+)\)$")


def _new_take(text: str) -> str:
    """The same description asked once more (a cached drawing is never reused for it)."""
    match = _TAKE.search(text or "")
    number = int(match.group(1)) + 1 if match else 2
    return f"{_TAKE.sub('', text or '')} (new take {number})"


def _unpin(place: dict) -> None:
    for key in ("image", "video"):
        place.pop(key, None)
    for part in (place.get("frames") or []) + (place.get("items") or []):
        if isinstance(part, dict):
            part.pop("image", None)


def _redo(shot: dict, pictures: List[dict], describe: str, query: str) -> None:
    """Make a shot again: its pictures are unpinned, its web pictures avoided and its drawings asked anew."""
    _unpin(shot)
    urls = [p.get("url") for p in pictures if p.get("url")]
    if urls:
        shot["avoid"] = list(dict.fromkeys(list(shot.get("avoid") or []) + urls))[-20:]
    if query.strip():
        shot["query"] = query.strip()[:120]
    frames = [f for f in shot.get("frames") or [] if isinstance(f, dict)]
    parts = [p.strip() for p in (describe or "").split(" / ") if p.strip()]
    if frames:
        if parts and parts != [f.get("draw", "") for f in frames]:
            shot["frames"] = [dict(frames[n] if n < len(frames) else {}, draw=part[:300]) for n, part in enumerate(parts[:4])]
        else:
            for frame in frames:
                frame["draw"] = _new_take(frame.get("draw", ""))
    elif describe.strip() and describe.strip() != shot.get("draw", ""):
        shot["draw"] = describe.strip()[:300]
    elif shot.get("draw"):
        shot["draw"] = _new_take(shot["draw"])


def apply_review(task_id: str, decisions: Dict[str, dict]) -> str:
    """Write the reviewed plan of a render prepared with --review and return its path.

    ``decisions`` is {shot id: {"action": keep | redo | photo | remove,
    "describe": new description, "query": new search}}: kept pictures stay
    pinned, redone ones are made again (from the new description, or as a
    new take), "photo" turns a drawn shot into a search for a real archive
    picture, and removed shots leave the plan (the shot before stays longer).
    """
    review = load_review(task_id)
    if review is None:
        raise ValueError(f"no review for task {task_id}")
    folder = Path(task_outputs(task_id)["folder"])
    plan_file = Path(review.get("plan_file") or folder / "edit-plan.json")
    # Decisions always apply to the plan as it was reviewed, so saving twice changes nothing more.
    data = _read_json(Path(review.get("reviewed_plan") or folder / "review-plan.json")) or _read_json(plan_file)
    if data is None or not isinstance(data.get("segments"), list):
        raise ValueError(f"the plan of task {task_id} cannot be read")
    shots = {shot["id"]: shot for shot in review["shots"] if isinstance(shot, dict) and "id" in shot}
    removed: Dict[int, set] = {}
    for shot_id, decision in decisions.items():
        listed = shots.get(shot_id)
        action = (decision or {}).get("action", "keep")
        if listed is None or action not in REVIEW_ACTIONS or action == "keep":
            continue
        segment = data["segments"][listed["segment"]] if 0 <= listed["segment"] < len(data["segments"]) else None
        if segment is None:
            continue
        describe = str(decision.get("describe") or "")
        query = str(decision.get("query") or "")
        if listed["shot"] < 0:  # the section's card: its photo can only be searched again
            opener = segment.setdefault("opener", {"query": listed.get("query") or listed.get("chapter", "")})
            if action in ("redo", "photo"):
                _redo(opener, listed.get("pictures") or [], "", query)
            continue
        planned = segment.get("shots") or []
        if not 0 <= listed["shot"] < len(planned):
            continue
        shot = planned[listed["shot"]]
        if action == "remove":
            removed.setdefault(listed["segment"], set()).add(listed["shot"])
        elif action == "redo":
            _redo(shot, listed.get("pictures") or [], describe, query)
        elif action == "photo":
            frames = [f.get("draw", "") for f in shot.get("frames") or [] if isinstance(f, dict)]
            draw = shot.get("draw") or (frames[0] if frames else "") or listed.get("describe", "")
            search = query.strip() or shot.get("query") or listed.get("query") or describe or draw
            planned[listed["shot"]] = {
                "type": "archive", "at": shot.get("at") or (shot.get("items") or [{}])[0].get("at", ""),
                "query": search[:120], "query_local": "", "caption": shot.get("text", "")[:60], "draw": draw[:300],
                **({"camera": shot["camera"]} if shot.get("camera") else {}),
                **({"avoid": shot["avoid"]} if shot.get("avoid") else {}),
            }
    for index, positions in removed.items():
        segment = data["segments"][index]
        segment["shots"] = [shot for n, shot in enumerate(segment.get("shots") or []) if n not in positions]
    _write_json(plan_file, data)
    return str(plan_file)


def clean_plan(data: dict) -> dict:
    """An edited plan made valid again: a storyboard (doodle look) or an edit plan (footage look), with its bible."""
    from app.services import llm

    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ValueError('the plan needs a "segments" list')
    segments = data["segments"]
    names: set = set()

    def collect(value) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("expression", "pose") and isinstance(item, str) and item:
                    names.add(item)
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(segments)
    if any(isinstance(entry, dict) and "shots" in entry for entry in segments):
        clean = {"segments": llm.normalize_storyboard(data, len(segments), sorted(names))}
    else:
        clean = {"segments": llm.normalize_edit_plan(data, len(segments), sorted(names))}
    if isinstance(data.get("bible"), dict):
        clean["bible"] = data["bible"]
    return clean


def youtube_description(title: str, intro: str, chapters: str, credits: str, hashtags: Sequence[str] = ()) -> str:
    """A ready-to-paste YouTube description: hook, chapters, credits and hashtags."""
    parts = [p for p in (title.strip(), intro.strip()) if p]
    if chapters.strip():
        parts.append("Capítulos:\n" + chapters.strip())
    if credits.strip():
        parts.append(credits.strip())
    if hashtags:
        parts.append(" ".join(h if h.startswith("#") else f"#{h}" for h in hashtags))
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Voices, settings and checks
# ---------------------------------------------------------------------------


def voice_groups() -> Dict[str, List[str]]:
    from app.services import voice, voice_google

    edge = [v for v in voice.get_all_azure_voices(["es-"]) if "-V2" not in v]
    return {
        "Recomendadas (español, voz joven masculina)": list(voice_google.SPANISH_MALE_SHORTLIST),
        "Gemini TTS (muy natural, ~US$0,015/min)": voice.get_gemini_voices(),
        "Google Cloud Chirp 3 HD (~US$0,03/min, nivel gratuito)": voice_google.get_gcloud_voices(),
        "Edge (gratis)": edge,
    }


SETTINGS_GROUPS = (
    ("Inteligencia artificial (Google)", ("llm_provider", "gemini_use_vertexai", "gemini_vertex_project", "gemini_vertex_location",
                                         "gemini_api_key", "gemini_model_name")),
    ("Modelos de voz e imagen", ("gemini_tts_model", "gemini_vision_model", "gemini_image_model", "gemini_video_model",
                                 "gemini_video_location", "gcloud_tts_api_key", "gcloud_tts_pitch", "api_key")),
    ("Imágenes, videos e investigación", ("pexels_api_keys", "pixabay_api_keys", "youtube_api_key", "ffmpeg_path")),
)
SETTINGS_FIELDS: Tuple[Tuple[str, str, str, str, str], ...] = (
    # (section, key, label, kind, help)
    ("app", "llm_provider", "Proveedor del LLM", "provider", "gemini para usar Gemini (Vertex AI o API key)"),
    ("app", "gemini_use_vertexai", "Usar Vertex AI", "bool", "Con gcloud auth application-default login; sin API key"),
    ("app", "gemini_vertex_project", "Proyecto de Google Cloud", "text", "ID del proyecto (Vertex AI y Cloud TTS)"),
    ("app", "gemini_vertex_location", "Región de Vertex AI", "text", "us-central1 o global"),
    ("app", "gemini_api_key", "Gemini API key", "secret", "Solo si no usas Vertex AI"),
    ("app", "gemini_model_name", "Modelo de texto", "text", "Ej.: gemini-2.5-flash o gemini-2.5-pro"),
    ("app", "gemini_tts_model", "Modelo de voz Gemini", "text", "gemini-2.5-flash-preview-tts o gemini-2.5-pro-preview-tts"),
    ("app", "gemini_vision_model", "Modelo que revisa imágenes", "text", "Por defecto gemini-2.5-flash"),
    ("app", "gemini_image_model", "Modelo que dibuja", "text",
     "Vacío = según la calidad elegida en Estilo (gemini-3.1-flash-image o gemini-3-pro-image)"),
    ("app", "gemini_video_model", "Modelo de video (Veo)", "text", "Vacío = veo-3.1-fast-generate-001 (Vertex AI)"),
    ("app", "gemini_video_location", "Región de Veo", "text",
     "Vacío = la misma de Vertex AI; us-central1 si Veo no está disponible en tu región (p. ej. global)"),
    ("app", "gcloud_tts_api_key", "Cloud TTS API key", "secret", "Opcional: si no, usa las credenciales de Vertex AI"),
    ("app", "gcloud_tts_pitch", "Tono Cloud TTS (semitonos)", "float", "Solo voces Neural2/Studio"),
    ("app", "pexels_api_keys", "Pexels API keys", "list", "Videos de fondo y fotos; separa varias con comas"),
    ("app", "pixabay_api_keys", "Pixabay API keys", "list", "Videos de fondo e ilustraciones"),
    ("app", "youtube_api_key", "YouTube Data API key", "secret", "Para la investigación de canales"),
    ("app", "ffmpeg_path", "Ruta de ffmpeg", "text", "Vacío para usar el del sistema (/usr/bin/ffmpeg)"),
    ("elevenlabs", "api_key", "ElevenLabs API key", "secret", "Voces elevenlabs:<voice_id>"),
)


def llm_providers() -> List[str]:
    from app.models.llm_provider import LLM_PROVIDERS

    return sorted(LLM_PROVIDERS)


def _section(name: str) -> dict:
    from app.config import config

    return getattr(config, name) if name != "app" else config.app


def read_settings() -> Dict[str, object]:
    values = {}
    for section, key, _, kind, _ in SETTINGS_FIELDS:
        value = _section(section).get(key, "")
        if kind == "list":
            value = ", ".join(value) if isinstance(value, list) else str(value or "")
        values[f"{section}.{key}"] = value
    return values


def write_settings(values: Dict[str, object]) -> None:
    from app.config import config

    for section, key, _, kind, _ in SETTINGS_FIELDS:
        name = f"{section}.{key}"
        if name not in values:
            continue
        value = values[name]
        if kind == "bool":
            value = bool(value)
        elif kind == "float":
            try:
                value = float(value or 0)
            except (TypeError, ValueError):
                value = 0.0
        elif kind == "list":
            value = [v.strip() for v in str(value or "").split(",") if v.strip()]
        else:
            value = str(value or "").strip()
        _section(section)[key] = value
    config.save_config()


def diagnostics() -> List[Tuple[str, bool, str]]:
    """(check, passed, detail) for what a render needs."""
    from app.config import config
    from app.services import gemini_media
    from app.utils import utils

    checks = []
    checks.append(("ffmpeg", bool(utils.check_ffmpeg_ready()), utils.get_ffmpeg_binary() or "no encontrado"))
    provider = str(config.app.get("llm_provider", "") or "")
    if provider == "gemini":
        llm_ready = gemini_media.enabled()
    elif provider in ("ollama", "g4f", ""):
        llm_ready = bool(provider)
    else:
        llm_ready = bool(str(config.app.get(f"{provider}_api_key", "") or "").strip())
    checks.append(("LLM", llm_ready, provider + ("" if llm_ready else ": falta la API key o la configuración") if provider else "sin configurar"))
    checks.append(("Gemini (imágenes y voces)", gemini_media.enabled(), "Vertex AI" if config.app.get("gemini_use_vertexai") else "API key"))
    adc = Path(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "") or Path.home() / ".config/gcloud/application_default_credentials.json")
    checks.append(("Credenciales de Google (ADC)", adc.is_file(), str(adc)))
    stock = bool(config.app.get("pexels_api_keys") or config.app.get("pixabay_api_keys"))
    checks.append(("Pexels o Pixabay", stock, "videos de fondo" if stock else "añade una API key"))
    nutria = Path(utils.resource_dir(os.path.join("characters", "nutria")))
    checks.append(("Nutria presentadora", nutria.is_dir(), str(nutria)))
    return checks


# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    os.replace(temporary, path)


if __name__ == "__main__":  # python -m app.services.studio run <job folder>
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        raise SystemExit(run_job(sys.argv[2]))
    raise SystemExit("usage: python -m app.services.studio run <job folder>")
