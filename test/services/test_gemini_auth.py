import os
import unittest
from unittest.mock import patch

from app.services import gemini_auth


class TestGeminiAuth(unittest.TestCase):
    def test_vertex_switch_reads_config_then_environment(self):
        with patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": "true"}):
            self.assertTrue(gemini_auth.use_vertexai({}))
            self.assertTrue(gemini_auth.use_vertexai({"gemini_use_vertexai": ""}))
            # An explicit setting wins over the environment.
            self.assertFalse(gemini_auth.use_vertexai({"gemini_use_vertexai": False}))
            self.assertFalse(gemini_auth.use_vertexai({"gemini_use_vertexai": "false"}))
        with patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": ""}):
            self.assertFalse(gemini_auth.use_vertexai({}))
            self.assertTrue(gemini_auth.use_vertexai({"gemini_use_vertexai": "yes"}))

    def test_vertex_client_options(self):
        config = {"gemini_use_vertexai": True, "gemini_vertex_project": " p1 ", "gemini_vertex_location": "global"}
        self.assertEqual(
            gemini_auth.client_kwargs(config, "ignored-key"),
            {"vertexai": True, "project": "p1", "location": "global"},
        )
        env = {"GOOGLE_CLOUD_PROJECT": "p2", "GOOGLE_CLOUD_LOCATION": "europe-west4"}
        with patch.dict(os.environ, env):
            self.assertEqual(
                gemini_auth.client_kwargs({"gemini_use_vertexai": True}),
                {"vertexai": True, "project": "p2", "location": "europe-west4"},
            )
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "", "GOOGLE_CLOUD_LOCATION": ""}):
            self.assertEqual(
                gemini_auth.client_kwargs({"gemini_use_vertexai": True, "gemini_vertex_project": "p3"})["location"],
                gemini_auth.DEFAULT_VERTEX_LOCATION,
            )
            with self.assertRaisesRegex(ValueError, "gemini_vertex_project"):
                gemini_auth.client_kwargs({"gemini_use_vertexai": True})

    def test_api_key_mode(self):
        with patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": ""}):
            self.assertEqual(gemini_auth.client_kwargs({}, " key "), {"api_key": "key"})
            with self.assertRaisesRegex(ValueError, "gemini_use_vertexai"):
                gemini_auth.client_kwargs({}, "")


if __name__ == "__main__":
    unittest.main()
