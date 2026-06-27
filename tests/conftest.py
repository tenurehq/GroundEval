import pathlib
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def isolate_filesystem(tmp_path, monkeypatch):
    """
    Force every test to operate inside a fresh temporary directory.

    - Changes the working directory to `tmp_path` for the duration of the test,
      so relative paths resolve inside the sandbox automatically.
    - Patches `builtins.open` to raise PermissionError if a test tries to open
      an absolute path outside the sandbox in any write/create/append mode.
    - pytest handles tmp_path creation and cleanup automatically; no manual
      tempfile.mkdtemp or shutil.rmtree needed.

    Yields the sandbox directory as a pathlib.Path so fixtures/tests that
    explicitly need the path can request it:

        def test_something(isolate_filesystem):
            p = isolate_filesystem / "data.txt"
            p.write_text("hello")
    """
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
