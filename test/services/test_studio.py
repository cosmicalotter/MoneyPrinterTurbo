import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import list_video as list_video_cli
from app.config import config
from app.services import studio
from app.utils import utils

SCRIPT = {
    "title": "La electricidad explicada para nutrias",
    "intro": "¿Qué es la electricidad?",
    "items": [
        {"name": "La carga eléctrica", "text": "Todo está hecho de átomos. " * 10, "image_term": "atoms"},
        {"name": "Voltaje y corriente", "text": "El voltaje empuja a los electrones. " * 10, "image_term": "battery"},
    ],
    "outro": "¿Qué otro tema quieres?",
}


class _StudioCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        real = utils.storage_dir

        def storage_dir(sub_dir="", create=False):
            path = os.path.join(self.root, sub_dir) if sub_dir else self.root
            if create:
                os.makedirs(path, exist_ok=True)
            return path

        patcher = patch.object(utils, "storage_dir", side_effect=storage_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.real_storage = real


class TestArgv(unittest.TestCase):
    def test_settings_become_list_video_arguments(self):
        settings = studio.RenderSettings(
            voice_name="gcloud:es-US-Chirp3-HD-Charon", voice_rate=1.1, host_presence="normal", openers=False,
            sound_effects=True, sfx_volume=0.5, music=True, subtitles=True, scene_color="#FFEEEE", lip_sync=True,
            also={"en-US": "gemini:Puck-Upbeat", "pt-BR": ""}, progress_bar=True, numbers=False, picture_check=False,
        )
        argv = studio.build_argv(settings, script_file="/p/script.json", edit_plan="/p/plan.json", task_id="0f8fad5b-d9cb-469f-a165-70867728950e")
        joined = " ".join(argv)
        for expected in (
            "--script /p/script.json", "--voice-name gcloud:es-US-Chirp3-HD-Charon", "--voice-rate 1.1",
            "--host-presence normal", "--no-openers", "--sfx-volume 0.5", "--bgm-type random", "--subtitle-enabled",
            "--scene-color #FFEEEE", "--lip-sync", "--edit-plan /p/plan.json", "--also-in en-US,pt-BR",
            "--also-voice en-US=gemini:Puck-Upbeat", "--progress-bar", "--no-numbers", "--no-picture-check", "--task-id 0f8fad5b-d9cb-469f-a165-70867728950e",
            "--voice-style calmado", "--format story", "--look doodle", "--canvas-color #F4C24F", "--logo nutria",
            "--pause 0.5", "--max-drawings 160",
        ):
            self.assertIn(expected, joined)
        self.assertNotIn("--items", argv)  # only for a subject
        self.assertNotIn("pt-BR=", joined)
        # Every argument is understood by list_video.py.
        parser = list_video_cli.build_parser()
        _, forwarded = parser.parse_known_args(argv)
        self.assertEqual(list_video_cli._find_unsupported_options(forwarded), [])

        footage = studio.build_argv(studio.RenderSettings(look="footage", voice_polish=False, boil=False, logo="", script_format="list"),
                                    script_file="/s.json")
        self.assertIn("--no-voice-polish", footage)
        self.assertNotIn("--canvas-color", footage)
        self.assertNotIn("--no-boil", footage)
        self.assertNotIn("--logo", footage)
        self.assertEqual(footage[footage.index("--format") + 1], "list")
        self.assertIn("--no-boil", studio.build_argv(studio.RenderSettings(boil=False), script_file="/s.json"))
        quiet = studio.build_argv(studio.RenderSettings(sound_effects=False), subject="La luz", script_only=True, output="/o.json")
        self.assertEqual(quiet[:2], ["--subject", "La luz"])
        for expected in ("--no-sfx", "--script-only", "--no-subtitle-enabled", "--words-per-item"):
            self.assertIn(expected, quiet)
        with self.assertRaises(ValueError):
            studio.build_argv(studio.RenderSettings())
        with self.assertRaises(ValueError):
            studio.build_argv(studio.RenderSettings(), subject="x", script_file="y")

    def test_settings_round_trip_ignores_unknown_keys(self):
        settings = studio.RenderSettings.from_dict({"aspect": "9:16", "bogus": 1, "also": {"en-US": "v"}})
        self.assertEqual((settings.aspect, settings.also), ("9:16", {"en-US": "v"}))
        self.assertEqual(studio.RenderSettings.from_dict(settings.to_dict()), settings)
        self.assertEqual(studio.RenderSettings.from_dict(None), studio.RenderSettings())


class TestProjects(_StudioCase):
    def test_create_list_and_save(self):
        slug = studio.create_project("La electricidad: ¡explicada!")
        self.assertEqual(slug, "la-electricidad-explicada")
        self.assertEqual(studio.create_project("La electricidad: ¡explicada!"), "la-electricidad-explicada-2")
        studio.save_script(slug, SCRIPT)
        self.assertEqual(studio.load_script(slug)["items"][1]["name"], "Voltaje y corriente")
        with self.assertRaises(ValueError):
            studio.save_script(slug, {"title": "", "items": []})
        # The editor's blank placeholder section is skipped; half-written ones are explained.
        blank = dict(SCRIPT, items=[{"name": "", "text": "", "image_term": ""}] + SCRIPT["items"])
        self.assertEqual(len(studio.validate_script(blank)[0]["items"]), 2)
        clean, error = studio.validate_script(dict(SCRIPT, items=[{"name": "Voltaje", "text": " "}]))
        self.assertIsNone(clean)
        self.assertEqual(error, 'la sección 1 no tiene narración ("text")')
        self.assertIn('no hay secciones ("items")', studio.validate_script(dict(SCRIPT, items=[{"name": "", "text": ""}]))[1])
        self.assertIn("objeto JSON", studio.validate_script([1])[1])
        settings = studio.load_settings(slug)
        settings.aspect = "9:16"
        studio.save_settings(slug, settings)
        self.assertEqual(studio.load_settings(slug).aspect, "9:16")
        projects = studio.list_projects()
        self.assertEqual({p["slug"] for p in projects}, {slug, f"{slug}-2"})
        self.assertTrue(next(p for p in projects if p["slug"] == slug)["has_script"])
        self.assertEqual(studio.word_count(SCRIPT), 4 + 50 + 60 + 4)
        self.assertGreater(studio.estimated_minutes(SCRIPT), 0.5)
        with self.assertRaises(ValueError):
            studio.project_dir("../etc")
        self.assertEqual(studio.slugify("¿?"), "proyecto")

    def test_outputs_and_description(self):
        folder = Path(self.root, "tasks", "t1")
        folder.mkdir(parents=True)
        (folder / "final-1.mp4").write_bytes(b"mp4")
        (folder / "chapters.txt").write_text("00:00 Intro\n00:20 1. La carga", encoding="utf-8")
        (folder / "credits.txt").write_text("Imágenes: Pexels", encoding="utf-8")
        (folder / "edit-plan.json").write_text("{}", encoding="utf-8")
        outputs = studio.task_outputs("t1")
        self.assertEqual([os.path.basename(v) for v in outputs["videos"]], ["final-1.mp4"])
        self.assertTrue(outputs["plan"].endswith("edit-plan.json"))
        self.assertEqual(outputs["script"], "")
        description = studio.youtube_description("Título", "Gancho", outputs["chapters"], outputs["credits"], ["ciencia", "#nutrias"])
        self.assertEqual(description, "Título\n\nGancho\n\nCapítulos:\n00:00 Intro\n00:20 1. La carga\n\nImágenes: Pexels\n\n#ciencia #nutrias\n")
        with self.assertRaises(ValueError):
            studio.task_outputs("../x")


class TestJobs(_StudioCase):
    LOG = (
        "2026 | INFO | generating list script\n"
        "2026 | INFO | list script generated\n"
        "2026 | INFO | list video narration [2/4] 1. A\n"
        "2026 | INFO | edit plan generated\n"
        "2026 | INFO | list video segment [3/4] 2. B\n"
        "2026 | ERROR | x - Gemini TTS is not configured\n"
        '{"task_id": "t1", "result": {"videos": ["v.mp4"]}}\n'
    )

    def test_progress_summary_and_errors(self):
        fraction, stage = studio.progress_from_log(self.LOG)
        self.assertAlmostEqual(fraction, 0.38 + 0.57 * 2 / 4)
        self.assertEqual(stage, "Montando la sección 3 de 4")
        self.assertEqual(studio.progress_from_log(""), (0.0, "Preparando"))
        self.assertEqual(studio.progress_from_log("list video finished: v.mp4")[0], 1.0)
        self.assertEqual(studio.parse_summary(self.LOG)["task_id"], "t1")
        self.assertIsNone(studio.parse_summary("no json"))
        self.assertEqual(len(studio.errors_from_log(self.LOG)), 1)

    def test_a_real_background_job(self):
        slug = studio.create_project("Prueba")
        job_id = studio.start_job(["--help"], label="Ayuda", project=slug, tool="research.py")
        for _ in range(120):
            info = studio.job_info(job_id)
            if info["state"] != "running":
                break
            time.sleep(0.5)
        self.assertEqual(info["state"], "done", info["log"][-500:])
        self.assertIn("usage", info["log"].lower())
        self.assertEqual(info["tool"], "research.py")
        self.assertEqual(info["progress"], 1.0)
        self.assertEqual([j["id"] for j in studio.list_jobs(project=slug)], [job_id])
        self.assertEqual(studio.project_meta(slug)["renders"], [])  # research is not a render
        self.assertFalse(studio.cancel_job(job_id))
        with self.assertRaises(ValueError):
            studio.start_job([], tool="rm")
        with self.assertRaises(ValueError):
            studio.job_info("../x")

    def test_runner_writes_how_it_ended(self):
        folder = studio.studio_dir("jobs", "1-1")
        (folder / "job.json").write_text(json.dumps({"argv": ["--bogus-option"], "tool": "voice_lab.py"}), encoding="utf-8")
        (folder / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
        with patch.object(studio.subprocess, "call", return_value=2) as call:
            self.assertEqual(studio.run_job(str(folder)), 2)
        self.assertTrue(call.call_args.args[0][3].endswith("voice_lab.py"))
        self.assertEqual(json.loads((folder / "status.json").read_text())["state"], "failed")
        info = studio.job_info("1-1")
        self.assertEqual(info["state"], "failed")

    def test_cancel_a_running_job(self):
        folder = studio.studio_dir("jobs", "2-2")
        (folder / "job.json").write_text("{}", encoding="utf-8")
        (folder / "status.json").write_text(json.dumps({"state": "running", "pid": 424242}), encoding="utf-8")
        with patch.object(studio.os, "killpg") as kill:
            self.assertTrue(studio.cancel_job("2-2"))
        kill.assert_called_once()
        self.assertEqual(studio.job_info("2-2")["state"], "cancelled")


class TestSettingsAndChecks(_StudioCase):
    def test_read_write_settings(self):
        with patch.dict(config.app, {"pexels_api_keys": ["a", "b"], "gemini_use_vertexai": False}), patch.object(config, "save_config") as save:
            values = studio.read_settings()
            self.assertEqual(values["app.pexels_api_keys"], "a, b")
            values.update({"app.pexels_api_keys": "x, y,", "app.gemini_use_vertexai": 1, "app.gcloud_tts_pitch": "bad",
                           "app.gemini_vertex_project": " mi-proyecto "})
            studio.write_settings(values)
            self.assertEqual(config.app["pexels_api_keys"], ["x", "y"])
            self.assertIs(config.app["gemini_use_vertexai"], True)
            self.assertEqual(config.app["gcloud_tts_pitch"], 0.0)
            self.assertEqual(config.app["gemini_vertex_project"], "mi-proyecto")
            save.assert_called_once()
        grouped = {key for _, keys in studio.SETTINGS_GROUPS for key in keys}
        self.assertEqual(grouped, {key for _, key, _, _, _ in studio.SETTINGS_FIELDS})
        self.assertIn("gemini", studio.llm_providers())

    def test_diagnostics_and_voices(self):
        with patch.dict(config.app, {"llm_provider": "moonshot", "moonshot_api_key": ""}):
            checks = dict((name, ok) for name, ok, _ in studio.diagnostics())
        self.assertFalse(checks["LLM"])
        self.assertIn("Nutria presentadora", checks)
        groups = studio.voice_groups()
        self.assertIn("gemini:Puck-Upbeat", groups["Recomendadas (español, voz joven masculina)"])
        self.assertTrue(any(v.startswith("gcloud:") for g in groups.values() for v in g))


class TestStudioApp(_StudioCase):
    PAGE = """
import sys
sys.path.insert(0, {folder!r})
import Studio
Studio.sidebar()
Studio.{page}()
"""

    def test_app_and_every_page_render(self):
        from streamlit.testing.v1 import AppTest

        slug = studio.create_project("La electricidad")
        studio.save_script(slug, SCRIPT)
        task = Path(self.root, "tasks", "t1")
        task.mkdir(parents=True)
        (task / "edit-plan.json").write_text(json.dumps({"segments": [{"index": 0, "scenes": [{"type": "stat"}], "beats": []}]}), encoding="utf-8")
        (task / "chapters.txt").write_text("00:00 Intro", encoding="utf-8")
        job = studio.studio_dir("jobs", "3-3")
        (job / "job.json").write_text(json.dumps({"argv": [], "project": slug, "label": "Video", "tool": "list_video.py"}), encoding="utf-8")
        (job / "status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")
        (job / "log.txt").write_text('{"task_id": "t1", "result": {"videos": []}}\n', encoding="utf-8")

        app = AppTest.from_file(str(Path(studio.ROOT, "studio", "Studio.py")), default_timeout=60)
        app.run()
        self.assertFalse(app.exception, app.exception)
        self.assertIn("Proyectos", [t.value for t in app.title])
        self.assertEqual(app.session_state["project"], slug)
        folder = str(Path(studio.ROOT, "studio"))
        for page in ("page_script", "page_voice", "page_style", "page_render", "page_results", "page_plan", "page_research", "page_settings"):
            page_app = AppTest.from_string(self.PAGE.format(folder=folder, page=page), default_timeout=60)
            page_app.session_state["project"] = slug
            page_app.run()
            self.assertFalse(page_app.exception, (page, page_app.exception))
            self.assertTrue(page_app.title, page)


if __name__ == "__main__":
    unittest.main()
