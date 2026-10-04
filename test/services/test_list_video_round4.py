import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import list_video as list_video_cli
from app.config import config as app_config
from app.models.schema import ListVideoItem, ListVideoScript
from app.services import list_video
from app.services import list_video_editor as editor
from app.services import list_video_fx as fx
from app.services import list_video_host as host
from app.utils import utils

RESOURCES = Path(__file__).parent.parent / "resources"
FONT = str(Path(utils.font_dir()) / "BeVietnamPro-Bold.ttf")
NAMES = ["explicando", "neutral", "feliz", "pensando", "saludando", "senalando", "sorprendido", "preocupado"]


def _icon(path, color=(220, 60, 60, 255), size=120):
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((10, 10, size - 10, size - 10), fill=color, outline=(0, 0, 0, 255), width=6)
    image.save(path)
    return path


def _narration(seconds, frames):
    return editor.Narration(pcm=b"", speech_seconds=seconds, frames=frames)


def _segments(text, names=("Ojos",)):
    return list_video.build_segments(
        ListVideoScript(
            title="T", intro="Hola nutrias.", items=[ListVideoItem(name=n, text=text) for n in names], outro="Chao."
        )
    )


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.temp_dir, name)


class TestChapterChip(unittest.TestCase):
    CHIP = fx.Overlay("chip.png", x="20-(300+44)*(1-(1-pow(1-clip((t-0.120)/0.450,0,1),3)))", y="20")

    def test_chip_hides_while_scenes_fill_the_screen(self):
        spans = editor.chip_spans(self.CHIP, [(4.0, 9.0), (14.0, 18.0)], 25.0, 344)
        self.assertEqual([(round(o.start, 2), round(o.end, 2)) for o in spans], [(0.0, 3.95), (9.05, 13.95), (18.05, 25.0)])
        first, middle, last = spans
        self.assertTrue(first.x.startswith(self.CHIP.x))  # the first entrance keeps its own slide
        self.assertIn("clip((t-3.650)/0.3,0,1)", first.x)  # and it slides out before the scene
        self.assertTrue(middle.x.startswith("20-344*(1-"))  # slides back in after the scene
        self.assertTrue(last.x.endswith("-0"))  # stays until the cut
        self.assertTrue(all(o.y == "20" and o.source == "chip.png" for o in spans))

    def test_short_gaps_and_scenes_at_the_edges(self):
        spans = editor.chip_spans(self.CHIP, [(0.0, 5.0), (5.8, 9.0)], 12.0, 344)
        self.assertEqual([(round(o.start, 2), round(o.end, 2)) for o in spans], [(9.05, 12.0)])
        self.assertEqual(editor.chip_spans(self.CHIP, [(0.0, 12.0)], 12.0, 344), [])
        alone = editor.chip_spans(self.CHIP, [], 12.0, 344)
        self.assertEqual(len(alone), 1)
        self.assertEqual((alone[0].start, alone[0].end), (0.0, 12.0))


class TestHostPresence(unittest.TestCase):
    def _share(self, presence):
        infos = [host.SegmentInfo(kind="intro", duration=6.0, expression="feliz")]
        infos += [host.SegmentInfo(kind="item", duration=24.0, expression="explicando") for _ in range(6)]
        infos += [host.SegmentInfo(kind="outro", duration=6.0, expression="feliz")]
        plan = host.plan_host(infos, NAMES, presence=presence)
        items = plan[1:-1]
        return sum(w.end - w.start for p in items for w in p.windows) / (24.0 * 6)

    def test_less_host_by_default(self):
        low, normal, high = self._share("low"), self._share("normal"), self._share("high")
        self.assertLess(low, normal)
        self.assertLess(normal, high)
        self.assertLess(low, 0.3)
        self.assertEqual(host.HOST_PRESENCES, ("low", "normal", "high"))
        # Intro and outro always keep the host, and unknown values behave like "normal".
        self.assertEqual(host.segment_mode("intro", 0, 0, "low"), "full")
        self.assertEqual(host.segment_mode("item", 2, 0, "nope"), host.ITEM_PATTERN[2])

    def test_lead_is_shorter_with_low_presence(self):
        info = host.SegmentInfo(kind="item", duration=20.0, expression="explicando", mode="lead")
        low = host.plan_host([info], NAMES, presence="low")[0].windows[0]
        normal = host.plan_host([info], NAMES, presence="normal")[0].windows[0]
        self.assertLess(low.end, normal.end)


class TestSoundVolume(_TempDirCase):
    TEXT = " ".join(f"palabra{i}" for i in range(60))

    def _edit(self, **options):
        assets = os.path.join(self.temp_dir, "assets")
        os.makedirs(os.path.join(assets, "sfx"), exist_ok=True)
        narrations = [_narration(1.0, 45), _narration(24.0, 730), _narration(1.0, 45)]
        theme = fx.Theme(1280, 720, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        options.setdefault("subscribe", "none")
        options.setdefault("host", "none")
        with patch.object(editor.gemini_media, "enabled", return_value=False):
            ed = editor.Editor(editor.EditOptions(**options), theme, self.temp_dir, _segments(self.TEXT), narrations)
        ed.sfx = {"whoosh": "w.wav", "pop": "p.wav", "click": "c.wav"}
        ed.plan = [
            {"index": 0, "expression": "", "scenes": [], "beats": []},
            {"index": 1, "expression": "", "scenes": [], "beats": []},
            {"index": 2, "expression": "", "scenes": [], "beats": []},
        ]
        return ed.segment_edit(1, 1.5, show_titles=True)

    def test_effects_are_quieter_by_default_and_can_be_muted(self):
        default = self._edit()
        louder = self._edit(sfx_volume=1.0)
        muted = self._edit(sfx_volume=0.0)
        self.assertTrue(default.sounds)
        for (_, _, quiet), (_, _, loud) in zip(default.sounds, louder.sounds):
            self.assertAlmostEqual(quiet, loud * 0.65)
        self.assertEqual(muted.sounds, [])
        chips = [o for o in default.overlays if os.path.basename(o.source).startswith("chip-")]
        self.assertEqual(len(chips), 1)


class TestRound4Cli(unittest.TestCase):
    def setUp(self):
        ui_patch = patch.dict(app_config.ui, {}, clear=True)
        ui_patch.start()
        self.addCleanup(ui_patch.stop)
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.script = os.path.join(self.temp_dir, "s.json")
        Path(self.script).write_text(json.dumps({"title": "T", "items": [{"name": "A", "text": "B"}]}), encoding="utf-8")

    def _edit(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(list_video, "generate_list_video", return_value={}) as generate, redirect_stdout(stdout), redirect_stderr(stderr):
            list_video_cli.run(["--script", self.script, *args])
        return generate.call_args.kwargs["edit"]

    def test_presence_and_volume_flags(self):
        default = self._edit()
        self.assertEqual((default.host_presence, default.sfx_volume), ("low", 0.65))
        custom = self._edit("--host-presence", "high", "--sfx-volume", "5")
        self.assertEqual((custom.host_presence, custom.sfx_volume), ("high", 2.0))


if __name__ == "__main__":
    unittest.main()
