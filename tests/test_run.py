import json
from pathlib import Path

from groundeval.adapters import FileCorpusAdapter
import pytest

from groundeval.core import EvalQuestion
from groundeval.run import _build_tool_specs, _build_context


def test_build_tool_specs_has_required_tools():
    specs = _build_tool_specs({})
    names = {s["name"] for s in specs}
    assert "fetch_artifact" in names
    assert "search_artifacts" in names


def test_build_context_perspective(tmp_path: Path):
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "a.json").write_text(
        '{"id": "a", "timestamp": "2026-03-01T00:00:00", "text": "later"}'
    )
    (root / "b.json").write_text(
        '{"id": "b", "timestamp": "2026-01-01T00:00:00", "text": "earlier"}'
    )
    from groundeval.adapters import FileCorpusAdapter

    corpus = FileCorpusAdapter(root)

    q = EvalQuestion(
        question_id="q1",
        question_type="PERSPECTIVE",
        question_text="?",
        difficulty="easy",
        ground_truth={},
        actor_visible_artifacts=["a", "b"],
        as_of_time="2026-06-01T00:00:00",
    )
    ctx = _build_context(q, corpus, max_tokens=100000)
    assert "a" in ctx
    assert "b" in ctx


def test_build_context_silence_uses_search_space(tmp_path: Path):
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "s1.json").write_text('{"id": "s1", "timestamp": "2026-01-01T00:00:00"}')
    from groundeval.adapters import FileCorpusAdapter

    corpus = FileCorpusAdapter(root)

    q = EvalQuestion(
        question_id="q1",
        question_type="SILENCE",
        question_text="?",
        difficulty="easy",
        ground_truth={},
        expected_search_space=["s1"],
    )
    ctx = _build_context(q, corpus, max_tokens=100000)
    assert "s1" in ctx


def test_build_context_empty_artifacts():
    class EmptyCorp:
        def fetch(self, aid):
            return None

        def timestamp_of(self, aid):
            return None

    q = EvalQuestion(
        question_id="q1",
        question_type="PERSPECTIVE",
        question_text="?",
        difficulty="easy",
        ground_truth={},
        actor_visible_artifacts=[],
    )
    ctx = _build_context(q, EmptyCorp(), max_tokens=1000)
    assert ctx == "(no artifacts available)"


def test_build_context_token_limit_truncation(tmp_path: Path):
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "small.json").write_text(
        json.dumps({
            "id": "small",
            "timestamp": "2026-01-01T00:00:00",
            "text": "hi",
        })
    )
    (root / "huge.json").write_text(
        json.dumps({
            "id": "huge",
            "timestamp": "2026-01-01T00:00:00",
            "text": "word " * 5000,
        })
    )
    from groundeval.adapters import FileCorpusAdapter

    corpus = FileCorpusAdapter(root)

    q = EvalQuestion(
        question_id="q1",
        question_type="PERSPECTIVE",
        question_text="?",
        difficulty="easy",
        ground_truth={},
        actor_visible_artifacts=["small", "huge"],
    )
    ctx = _build_context(q, corpus, max_tokens=200)  # tight budget
    assert "(no artifacts available)" not in ctx
    assert "--- small ---" in ctx
    assert "--- huge ---" not in ctx


def test_build_context_silence_skips_bracket_hints(tmp_path: Path):
    root = tmp_path / "artifacts"
    root.mkdir()
    corpus = FileCorpusAdapter(root)

    q = EvalQuestion(
        question_id="q1",
        question_type="SILENCE",
        question_text="?",
        difficulty="easy",
        ground_truth={},
        expected_search_space=["[jira] query term", "real-id"],
    )
    ctx = _build_context(q, corpus, max_tokens=1000)
    assert "[jira] query term" not in ctx  # bracket hints are skipped
