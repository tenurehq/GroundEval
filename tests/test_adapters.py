import json
from pathlib import Path
import tempfile
from typing import Optional

import pytest

from groundeval.core import LogEvent
from groundeval.adapters import (
    EventLogPolicy,
    FileCorpusAdapter,
    NullCorpusAdapter,
    YamlAccessPolicy,
)


def test_file_corpus_adapter_flat(tmp_path: Path):
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "doc1.json").write_text(
        json.dumps({"id": "doc1", "timestamp": "2026-01-01T00:00:00", "text": "hi"})
    )
    adapter = FileCorpusAdapter(root)
    assert adapter.list_ids() == ["doc1"]
    doc = adapter.fetch("doc1")
    assert doc is not None
    assert doc["id"] == "doc1"


def test_file_corpus_adapter_nested_subsystem(tmp_path: Path):
    root = tmp_path / "artifacts"
    (root / "jira").mkdir(parents=True)
    (root / "jira" / "TICKET-1.json").write_text(
        json.dumps({"id": "TICKET-1", "timestamp": "2026-01-01T00:00:00"})
    )
    adapter = FileCorpusAdapter(root)
    assert adapter.subsystem_of("TICKET-1") == "jira"


def test_file_corpus_search_and_as_of(tmp_path: Path):
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "a.json").write_text(
        json.dumps({
            "id": "a",
            "timestamp": "2026-03-01T00:00:00",
            "subsystem": "email",
            "body": "hello world",
        })
    )
    (root / "b.json").write_text(
        json.dumps({
            "id": "b",
            "timestamp": "2026-01-01T00:00:00",
            "subsystem": "email",
            "body": "hello mars",
        })
    )
    adapter = FileCorpusAdapter(root)
    results = adapter.search("hello")
    assert len(results) == 2

    filtered = adapter.search("hello", as_of="2026-02-01T00:00:00")
    assert len(filtered) == 1
    assert filtered[0]["id"] == "b"


def test_null_corpus_adapter_returns_empty():
    n = NullCorpusAdapter()
    assert n.fetch("x") is None
    assert n.search("x") == []
    assert n.list_ids() == []


def test_yaml_access_policy_visible_artifacts():
    config = {
        "actors": {"alice": "engineer"},
        "roles": {"engineer": {"subsystems": ["jira"]}},
    }
    policy = YamlAccessPolicy(config)
    assert policy.role_for_actor("alice") == "engineer"
    assert policy.subsystems_for_role("engineer") == {"jira"}

    class MiniCorpus:
        def subsystem_of(self, artifact_id: str) -> Optional[str]:
            return {"j1": "jira", "s1": "slack"}.get(artifact_id)

    visible = policy.visible_artifacts("alice", ["j1", "s1"], corpus=MiniCorpus())
    assert visible == {"j1"}


def test_event_log_policy_direct_involvement():
    events = [
        LogEvent(
            id="e1",
            type="incident",
            timestamp="2026-01-01T00:00:00",
            actors=["alice", "bob"],
            artifact_ids={"jira": "J-1"},
        ),
        LogEvent(
            id="e2",
            type="note",
            timestamp="2026-01-02T00:00:00",
            actors=["carol"],
            artifact_ids={"jira": "J-2"},
        ),
    ]
    config = {
        "actors": {"alice": "engineer", "bob": "engineer", "carol": "engineer"},
        "roles": {"engineer": {"subsystems": ["jira"]}},
    }
    policy = EventLogPolicy(config, events)
    visible = policy.visible_artifacts("alice", ["J-1", "J-2"])
    assert "J-1" in visible
    assert "J-2" in visible


def test_event_log_policy_broadcast_event_type():
    events = [
        LogEvent(
            id="e1",
            type="incident_opened",
            timestamp="2026-01-01T00:00:00",
            actors=["system"],
            artifact_ids={"slack": "slack-1"},
        ),
    ]
    config = {
        "actors": {"alice": "engineer"},
        "roles": {
            "engineer": {"subsystems": [], "broadcast_event_types": ["incident_opened"]}
        },
    }
    policy = EventLogPolicy(config, events)
    visible = policy.visible_artifacts("alice", ["slack-1"])
    assert "slack-1" in visible


def test_event_log_policy_temporal_filter(monkeypatch):
    events = [
        LogEvent(
            id="e1",
            type="incident",
            timestamp="2026-01-05T00:00:00",
            actors=["alice"],
            artifact_ids={"jira": "J-1"},
        ),
    ]
    config = {
        "actors": {"alice": "engineer"},
        "roles": {"engineer": {"subsystems": ["jira"]}},
    }

    class MinCorpus:
        def timestamp_of(self, aid: str) -> Optional[str]:
            return "2026-01-05T00:00:00"

        def subsystem_of(self, aid: str) -> Optional[str]:
            return "jira"

    policy = EventLogPolicy(config, events)
    visible = policy.visible_artifacts(
        "alice", ["J-1"], as_of="2026-01-01T00:00:00", corpus=MinCorpus()
    )
    assert "J-1" not in visible

    visible2 = policy.visible_artifacts(
        "alice", ["J-1"], as_of="2026-01-10T00:00:00", corpus=MinCorpus()
    )
    assert "J-1" in visible2


def test_file_corpus_missing_artifact():
    with tempfile.TemporaryDirectory() as td:
        adapter = FileCorpusAdapter(td)
        assert adapter.fetch("missing") is None
        assert adapter.timestamp_of("missing") is None


def test_file_corpus_empty_directory():
    with tempfile.TemporaryDirectory() as td:
        adapter = FileCorpusAdapter(td)
        assert adapter.list_ids() == []
        assert adapter.search("anything") == []


def test_yaml_access_policy_no_roles():
    policy = YamlAccessPolicy({"actors": {"alice": "engineer"}})
    assert policy.subsystems_for_role("engineer") == set()
    assert policy.visible_artifacts("alice", ["a", "b"], corpus=None) == {"a", "b"}

    class Corp:
        def subsystem_of(self, artifact_id):
            return "slack"

    assert policy.visible_artifacts("alice", ["a", "b"], corpus=Corp()) == set()


def test_yaml_access_policy_unknown_actor():
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    assert policy.role_for_actor("nobody") is None
    assert policy.visible_artifacts("nobody", ["a"], corpus=None) == set()


def test_event_log_policy_exact_timestamp_boundary():
    """Artifact timestamp exactly at as_of should remain visible."""
    evt = LogEvent(
        id="e1",
        type="incident",
        timestamp="2026-01-15T09:00:00",
        actors=["alice"],
        artifact_ids={"jira": "J-1"},
    )
    config = {
        "actors": {"alice": "eng"},
        "roles": {"eng": {"subsystems": ["jira"]}},
    }

    class Corp:
        def timestamp_of(self, aid):
            return "2026-01-15T09:00:00"

        def subsystem_of(self, aid):
            return "jira"

    policy = EventLogPolicy(config, [evt])
    assert "J-1" in policy.visible_artifacts(
        "alice", ["J-1"], as_of="2026-01-15T09:00:00", corpus=Corp()
    )


def test_event_log_policy_empty_artifact_ids():
    evt = LogEvent(
        id="e1",
        type="t",
        timestamp="2026-01-01T00:00:00",
        actors=["alice"],
        artifact_ids={},
    )
    config = {
        "actors": {"alice": "eng"},
        "roles": {"eng": {"subsystems": ["jira"]}},
    }
    policy = EventLogPolicy(config, [evt])

    assert policy.visible_artifacts("alice", ["J-1"], corpus=None) == {"J-1"}

    class Corp:
        def subsystem_of(self, artifact_id):
            return "slack"

    assert policy.visible_artifacts("alice", ["J-1"], corpus=Corp()) == set()
