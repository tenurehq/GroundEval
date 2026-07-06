import pathlib
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def isolate_filesystem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    real_open = open

    def guarded_open(file, mode="r", *args, **kwargs):
        path = pathlib.Path(file)
        if path.is_absolute() and any(ch in mode for ch in ("w", "a", "x")):
            try:
                path.relative_to(tmp_path)
            except ValueError:
                raise PermissionError(
                    f"Test attempted to write outside the sandbox: {path}\n"
                    f"Sandbox root: {tmp_path}\n"
                    "Use relative paths or tmp_path to stay inside the sandbox."
                )
        return real_open(file, mode, *args, **kwargs)

    with patch("builtins.open", guarded_open):
        yield tmp_path
