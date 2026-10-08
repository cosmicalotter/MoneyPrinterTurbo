"""Shared pytest setup: every test gets its own caches of narrations, drawings and AI videos."""

import os

import pytest

from app.utils import utils

PRIVATE_CACHES = ("narration", "illustrations", "videos")


@pytest.fixture(autouse=True)
def _private_caches(tmp_path, monkeypatch):
    """A take or a drawing cached by one test (or by a real render) never answers for another test."""
    real = utils.storage_dir

    def storage_dir(sub_dir: str = "", create: bool = False):
        parts = os.path.normpath(sub_dir or "").split(os.sep)
        if len(parts) == 2 and parts[0] == "cache" and parts[1] in PRIVATE_CACHES:
            folder = tmp_path / "cache" / parts[1]
            if create:
                folder.mkdir(parents=True, exist_ok=True)
            return str(folder)
        return real(sub_dir, create)

    monkeypatch.setattr(utils, "storage_dir", storage_dir)
