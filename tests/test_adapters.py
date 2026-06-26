import json
import tempfile
from pathlib import Path

import pytest

from groundeval.adapters import (
    FileCorpusAdapter,
    InMemoryCorpusAdapter,
    NullCorpusAdapter,
    YamlAccessPolicy,
)


# ── FileCorpusAdapter ───────────────────────────────────────


def test_file_corpus_fetch_single_artifact():
    """Fetches a single artifact from a flat directory."""
    with tempfile.TemporaryDirectory() as tmp:
        art = Path(tmp) / "crm_account.json"
        art.write_text(
            json.dumps({
                "id": "crm_account",
                "subsystem": "crm",
                "account_status": "active",
            })
        )

        corpus = FileCorpusAdapter(tmp)
        doc = corpus.fetch("crm_account")
        assert doc is not None
        assert doc["id"] == "crm_account"
        assert doc["account_status"] == "active"


def test_file_corpus_fetch_missing():
    """Returns None for non-existent artifact."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus = FileCorpusAdapter(tmp)
        assert corpus.fetch("nonexistent") is None


def test_file_corpus_fetch_as_of_blocks_future():
    """as_of parameter blocks artifacts with later timestamps."""
    with tempfile.TemporaryDirectory() as tmp:
        art = Path(tmp) / "email.json"
        art.write_text(
            json.dumps({
                "id": "email",
                "timestamp": "2026-06-15T09:00:00",
                "subsystem": "email",
            })
        )

        corpus = FileCorpusAdapter(tmp)
        doc = corpus.fetch("email", as_of="2026-01-01T00:00:00")
        assert doc is None


def test_file_corpus_fetch_as_of_allows_past():
    """as_of parameter allows artifacts with earlier timestamps."""
    with tempfile.TemporaryDirectory() as tmp:
        art = Path(tmp) / "old.json"
        art.write_text(
            json.dumps({
                "id": "old",
                "timestamp": "2025-01-01T00:00:00",
                "subsystem": "crm",
            })
        )

        corpus = FileCorpusAdapter(tmp)
        doc = corpus.fetch("old", as_of="2026-01-01T00:00:00")
        assert doc is not None


def test_file_corpus_search_by_query():
    """Search finds artifacts matching query text."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.json").write_text(
            json.dumps({"id": "a", "subsystem": "crm", "name": "Acme Corp"})
        )
        (Path(tmp) / "b.json").write_text(
            json.dumps({"id": "b", "subsystem": "email", "name": "Beta Inc"})
        )
        (Path(tmp) / "c.json").write_text(
            json.dumps({"id": "c", "subsystem": "crm", "name": "Acme Logistics"})
        )

        corpus = FileCorpusAdapter(tmp)
        results = corpus.search("Acme")
        assert len(results) == 2
        ids = {r["id"] for r in results}
        assert ids == {"a", "c"}


def test_file_corpus_search_by_type():
    """Search filters by artifact_type (subsystem)."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.json").write_text(
            json.dumps({"id": "a", "subsystem": "crm", "name": "Acme"})
        )
        (Path(tmp) / "b.json").write_text(
            json.dumps({"id": "b", "subsystem": "email", "name": "Acme"})
        )

        corpus = FileCorpusAdapter(tmp)
        results = corpus.search("Acme", artifact_type="email")
        assert len(results) == 1
        assert results[0]["id"] == "b"


def test_file_corpus_search_limit():
    """Search respects the limit parameter."""
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(10):
            (Path(tmp) / f"doc_{i}.json").write_text(
                json.dumps({"id": f"doc_{i}", "subsystem": "crm", "text": "common"})
            )

        corpus = FileCorpusAdapter(tmp)
        results = corpus.search("common", limit=3)
        assert len(results) <= 3


def test_file_corpus_search_as_of():
    """Search filters by as_of timestamp."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "old.json").write_text(
            json.dumps({
                "id": "old",
                "subsystem": "crm",
                "timestamp": "2025-01-01T00:00:00",
            })
        )
        (Path(tmp) / "new.json").write_text(
            json.dumps({
                "id": "new",
                "subsystem": "crm",
                "timestamp": "2026-06-01T00:00:00",
            })
        )

        corpus = FileCorpusAdapter(tmp)
        results = corpus.search("crm", as_of="2025-06-01T00:00:00")
        ids = {r["id"] for r in results}
        assert "old" in ids
        assert "new" not in ids


def test_file_corpus_timestamp_of():
    """Returns timestamp of an artifact."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.json").write_text(
            json.dumps({"id": "a", "timestamp": "2026-03-15T10:00:00"})
        )

        corpus = FileCorpusAdapter(tmp)
        assert corpus.timestamp_of("a") == "2026-03-15T10:00:00"
        assert corpus.timestamp_of("missing") is None


def test_file_corpus_subsystem_of():
    """Returns subsystem from artifact or subsystem_map override."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.json").write_text(json.dumps({"id": "a", "subsystem": "crm"}))

        corpus = FileCorpusAdapter(tmp, subsystem_map={"b": "email"})
        assert corpus.subsystem_of("a") == "crm"
        assert corpus.subsystem_of("b") == "email"
        assert corpus.subsystem_of("missing") is None


def test_file_corpus_list_ids():
    """Lists all artifact IDs, optionally filtered by subsystem."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.json").write_text(json.dumps({"id": "a", "subsystem": "crm"}))
        (Path(tmp) / "b.json").write_text(json.dumps({"id": "b", "subsystem": "email"}))

        corpus = FileCorpusAdapter(tmp)
        all_ids = corpus.list_ids()
        assert set(all_ids) == {"a", "b"}

        crm_ids = corpus.list_ids(subsystem="crm")
        assert crm_ids == ["a"]

        assert corpus.list_ids(subsystem="jira") == []


def test_file_corpus_nested_directories():
    """Artifacts in nested directories are discovered with subsystem from parent dir."""
    with tempfile.TemporaryDirectory() as tmp:
        crm_dir = Path(tmp) / "crm"
        crm_dir.mkdir()
        (crm_dir / "account.json").write_text(
            json.dumps({"id": "account", "customer": "Acme"})
        )

        corpus = FileCorpusAdapter(tmp)
        doc = corpus.fetch("account")
        assert doc is not None
        assert doc["subsystem"] == "crm"


def test_file_corpus_caching():
    """Repeated fetches return cached documents."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.json").write_text(json.dumps({"id": "a", "value": 42}))

        corpus = FileCorpusAdapter(tmp)
        doc1 = corpus.fetch("a")
        doc2 = corpus.fetch("a")
        assert doc1 is doc2  # Same object from cache


# ── NullCorpusAdapter ───────────────────────────────────────


def test_null_corpus_always_returns_none():
    """All retrieval methods return None or empty."""
    corpus = NullCorpusAdapter()
    assert corpus.fetch("anything") is None
    assert corpus.fetch("anything", as_of="2026-01-01") is None
    assert corpus.search("query") == []
    assert corpus.search("query", artifact_type="crm") == []
    assert corpus.timestamp_of("x") is None
    assert corpus.subsystem_of("x") is None
    assert corpus.list_ids() == []
    assert corpus.list_ids(subsystem="crm") == []


# ── YamlAccessPolicy ────────────────────────────────────────


def test_yaml_policy_role_for_actor():
    """Maps actor ID to role name."""
    policy = YamlAccessPolicy({
        "actors": {"alice": "engineer", "bob": "sales"},
        "roles": {},
    })
    assert policy.role_for_actor("alice") == "engineer"
    assert policy.role_for_actor("bob") == "sales"
    assert policy.role_for_actor("unknown") is None


def test_yaml_policy_subsystems_for_role():
    """Returns subsystem set for a role."""
    policy = YamlAccessPolicy({
        "actors": {},
        "roles": {
            "engineer": {"subsystems": ["jira", "git", "slack"]},
            "sales": {"subsystems": ["crm", "email"]},
        },
    })
    assert policy.subsystems_for_role("engineer") == {"jira", "git", "slack"}
    assert policy.subsystems_for_role("sales") == {"crm", "email"}
    assert policy.subsystems_for_role("unknown") == set()


def test_yaml_policy_visible_artifacts():
    """Returns only artifacts whose subsystem the actor can access."""
    policy = YamlAccessPolicy({
        "actors": {"alice": "engineer"},
        "roles": {"engineer": {"subsystems": ["jira", "git"]}},
    })

    corpus = InMemoryCorpusAdapter([
        {"id": "a1", "subsystem": "jira"},
        {"id": "a2", "subsystem": "git"},
        {"id": "a3", "subsystem": "email"},
    ])

    visible = policy.visible_artifacts("alice", ["a1", "a2", "a3"], corpus=corpus)
    assert visible == {"a1", "a2"}


def test_yaml_policy_visible_artifacts_unknown_actor():
    """Actor not in config sees nothing."""
    policy = YamlAccessPolicy({
        "actors": {"alice": "engineer"},
        "roles": {"engineer": {"subsystems": ["jira"]}},
    })

    corpus = InMemoryCorpusAdapter([
        {"id": "a1", "subsystem": "jira"},
    ])

    visible = policy.visible_artifacts("bob", ["a1"], corpus=corpus)
    assert visible == set()


def test_yaml_policy_visible_artifacts_no_corpus():
    """Without corpus, returns empty set since no artifact-to-subsystem mapping exists."""
    policy = YamlAccessPolicy({
        "actors": {"alice": "engineer"},
        "roles": {"engineer": {"subsystems": ["jira"]}},
    })
    visible = policy.visible_artifacts("alice", ["a1", "a2", "a3"])
    assert visible == set()


def test_yaml_policy_visible_artifacts_none_subsystem():
    """Artifacts with no subsystem are visible regardless of role."""
    policy = YamlAccessPolicy({
        "actors": {"alice": "engineer"},
        "roles": {"engineer": {"subsystems": ["jira"]}},
    })

    corpus = InMemoryCorpusAdapter([
        {"id": "a1", "subsystem": "jira"},
        {"id": "a2"},  # no subsystem
    ])

    visible = policy.visible_artifacts("alice", ["a1", "a2"], corpus=corpus)
    assert visible == {"a1", "a2"}


# ── InMemoryCorpusAdapter ───────────────────────────────────


def test_inmemory_fetch():
    """Fetches artifact from in-memory store."""
    corpus = InMemoryCorpusAdapter([
        {"id": "a", "subsystem": "crm", "value": 42},
        {"id": "b", "subsystem": "email", "value": 99},
    ])
    doc = corpus.fetch("a")
    assert doc is not None
    assert doc["value"] == 42
    assert corpus.fetch("missing") is None


def test_inmemory_fetch_as_of():
    """as_of blocks future artifacts."""
    corpus = InMemoryCorpusAdapter([
        {"id": "a", "timestamp": "2026-06-15T00:00:00"},
    ])
    assert corpus.fetch("a", as_of="2025-01-01T00:00:00") is None
    assert corpus.fetch("a", as_of="2027-01-01T00:00:00") is not None


def test_inmemory_search():
    """Search matches against JSON content."""
    corpus = InMemoryCorpusAdapter([
        {"id": "a", "subsystem": "crm", "name": "Acme Corp"},
        {"id": "b", "subsystem": "email", "name": "Beta Inc"},
        {"id": "c", "subsystem": "crm", "name": "Acme Logistics"},
    ])
    results = corpus.search("Acme")
    assert len(results) == 2
    ids = {r["id"] for r in results}
    assert ids == {"a", "c"}


def test_inmemory_search_type_filter():
    """Search filters by artifact_type."""
    corpus = InMemoryCorpusAdapter([
        {"id": "a", "subsystem": "crm", "name": "Common"},
        {"id": "b", "subsystem": "email", "name": "Common"},
    ])
    results = corpus.search("Common", artifact_type="email")
    assert len(results) == 1
    assert results[0]["id"] == "b"


def test_inmemory_search_limit():
    """Search respects limit."""
    docs = [{"id": f"d{i}", "subsystem": "crm", "text": "x"} for i in range(10)]
    corpus = InMemoryCorpusAdapter(docs)
    results = corpus.search("x", limit=3)
    assert len(results) <= 3


def test_inmemory_timestamp_of():
    """Returns timestamp."""
    corpus = InMemoryCorpusAdapter([
        {"id": "a", "timestamp": "2026-01-15T09:00:00"},
    ])
    assert corpus.timestamp_of("a") == "2026-01-15T09:00:00"
    assert corpus.timestamp_of("missing") is None


def test_inmemory_subsystem_of():
    """Returns subsystem."""
    corpus = InMemoryCorpusAdapter([
        {"id": "a", "subsystem": "crm"},
        {"id": "b"},
    ])
    assert corpus.subsystem_of("a") == "crm"
    assert corpus.subsystem_of("b") is None
    assert corpus.subsystem_of("missing") is None


def test_inmemory_list_ids():
    """Lists all or filtered IDs."""
    corpus = InMemoryCorpusAdapter([
        {"id": "a", "subsystem": "crm"},
        {"id": "b", "subsystem": "email"},
        {"id": "c", "subsystem": "crm"},
    ])
    assert set(corpus.list_ids()) == {"a", "b", "c"}
    assert set(corpus.list_ids(subsystem="crm")) == {"a", "c"}
    assert corpus.list_ids(subsystem="jira") == []


def test_inmemory_uses_id_or_underscore_id():
    """Handles both 'id' and '_id' keys."""
    corpus = InMemoryCorpusAdapter([
        {"_id": "mongo_style", "value": 1},
        {"id": "regular", "value": 2},
    ])
    assert corpus.fetch("mongo_style") is not None
    assert corpus.fetch("regular") is not None
    assert "mongo_style" in corpus.list_ids()
    assert "regular" in corpus.list_ids()
