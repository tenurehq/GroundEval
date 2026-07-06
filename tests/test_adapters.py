import json

from groundeval.adapters import (
    FileCorpusAdapter,
    InMemoryCorpusAdapter,
    NullCorpusAdapter,
    YamlAccessPolicy,
)


def test_file_corpus_fetch_single_artifact(tmp_path):
    p = tmp_path / "email-001.json"
    p.write_text(json.dumps({"id": "email-001", "subsystem": "email", "value": 1}))

    corpus = FileCorpusAdapter(tmp_path)
    doc = corpus.fetch("email-001")
    assert doc is not None
    assert doc["id"] == "email-001"
    assert doc["value"] == 1


def test_file_corpus_fetch_missing(tmp_path):
    corpus = FileCorpusAdapter(tmp_path)
    assert corpus.fetch("missing") is None


def test_file_corpus_fetch_as_of_filters_future(tmp_path):
    p = tmp_path / "doc.json"
    p.write_text(
        json.dumps({
            "id": "doc",
            "timestamp": "2026-06-01T00:00:00",
            "subsystem": "crm",
        })
    )

    corpus = FileCorpusAdapter(tmp_path)
    assert corpus.fetch("doc", as_of="2026-01-01T00:00:00") is None
    assert corpus.fetch("doc", as_of="2027-01-01T00:00:00") is not None


def test_file_corpus_search_query_and_limit(tmp_path):
    for i in range(5):
        (tmp_path / f"a{i}.json").write_text(
            json.dumps({"id": f"a{i}", "subsystem": "crm", "name": "Acme"})
        )

    corpus = FileCorpusAdapter(tmp_path)
    results = corpus.search("Acme", limit=2)
    assert len(results) == 2


def test_file_corpus_search_artifact_type(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps({"id": "a", "subsystem": "crm", "name": "Acme"})
    )
    (tmp_path / "b.json").write_text(
        json.dumps({"id": "b", "subsystem": "email", "name": "Acme"})
    )

    corpus = FileCorpusAdapter(tmp_path)
    results = corpus.search("Acme", artifact_type="email")
    assert len(results) == 1
    assert results[0]["id"] == "b"


def test_file_corpus_nested_subsystem_inferred(tmp_path):
    jira = tmp_path / "jira"
    jira.mkdir()
    (jira / "TICKET-42.json").write_text(
        json.dumps({"id": "TICKET-42", "title": "Bug"})
    )

    corpus = FileCorpusAdapter(tmp_path)
    doc = corpus.fetch("TICKET-42")
    assert doc is not None
    assert doc["subsystem"] == "jira"


def test_file_corpus_list_ids_can_include_stem_and_id(tmp_path):
    p = tmp_path / "file_name.json"
    p.write_text(json.dumps({"id": "logical-id", "subsystem": "crm"}))

    corpus = FileCorpusAdapter(tmp_path)
    ids = corpus.list_ids()
    assert "file_name" in ids
    assert "logical-id" in ids


def test_file_corpus_timestamp_and_subsystem_lookup(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(
        json.dumps({
            "id": "a",
            "timestamp": "2026-01-01T00:00:00",
            "subsystem": "crm",
        })
    )

    corpus = FileCorpusAdapter(tmp_path)
    assert corpus.timestamp_of("a") == "2026-01-01T00:00:00"
    assert corpus.subsystem_of("a") == "crm"


def test_file_corpus_subsystem_map_override(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"id": "a"}))

    corpus = FileCorpusAdapter(tmp_path, subsystem_map={"a": "email"})
    assert corpus.subsystem_of("a") == "email"


def test_file_corpus_array_file_indexes_each_item(tmp_path):
    p = tmp_path / "batch.json"
    p.write_text(
        json.dumps([
            {"id": "a1", "subsystem": "crm", "name": "One"},
            {"id": "a2", "subsystem": "email", "name": "Two"},
        ])
    )

    corpus = FileCorpusAdapter(tmp_path)
    assert corpus.fetch("a1")["name"] == "One"
    assert corpus.fetch("a2")["name"] == "Two"


def test_inmemory_corpus_basic_behavior():
    corpus = InMemoryCorpusAdapter([
        {"id": "a", "subsystem": "crm", "name": "Acme"},
        {"_id": "b", "subsystem": "email", "name": "Beta"},
    ])

    assert corpus.fetch("a")["name"] == "Acme"
    assert corpus.fetch("b")["name"] == "Beta"
    assert corpus.fetch("missing") is None
    assert corpus.subsystem_of("a") == "crm"
    assert corpus.timestamp_of("a") is None
    assert set(corpus.list_ids()) == {"a", "b"}


def test_inmemory_search_filters():
    corpus = InMemoryCorpusAdapter([
        {"id": "a", "subsystem": "crm", "name": "Acme"},
        {"id": "b", "subsystem": "email", "name": "Acme"},
        {
            "id": "c",
            "subsystem": "crm",
            "timestamp": "2027-01-01T00:00:00",
            "name": "Acme",
        },
    ])

    assert len(corpus.search("Acme", artifact_type="crm")) == 2
    assert (
        len(corpus.search("Acme", artifact_type="crm", as_of="2026-01-01T00:00:00"))
        == 1
    )
    assert corpus.search("Acme", limit=0) == []


def test_null_corpus_returns_empty():
    corpus = NullCorpusAdapter()
    assert corpus.fetch("x") is None
    assert corpus.search("x") == []
    assert corpus.timestamp_of("x") is None
    assert corpus.subsystem_of("x") is None
    assert corpus.list_ids() == []


def test_yaml_access_policy_basic():
    policy = YamlAccessPolicy({
        "actors": {"alice": "engineer"},
        "roles": {"engineer": {"subsystems": ["jira", "git"]}},
    })

    assert policy.role_for_actor("alice") == "engineer"
    assert policy.role_for_actor("bob") is None
    assert policy.subsystems_for_role("engineer") == {"jira", "git"}
    assert policy.subsystems_for_role("missing") == set()


def test_yaml_access_policy_visible_artifacts():
    corpus = InMemoryCorpusAdapter([
        {"id": "a1", "subsystem": "jira"},
        {"id": "a2", "subsystem": "git"},
        {"id": "a3", "subsystem": "email"},
        {"id": "a4"},
    ])
    policy = YamlAccessPolicy({
        "actors": {"alice": "engineer"},
        "roles": {"engineer": {"subsystems": ["jira", "git"]}},
    })

    visible = policy.visible_artifacts("alice", ["a1", "a2", "a3", "a4"], corpus=corpus)
    assert visible == {"a1", "a2", "a4"}


def test_yaml_access_policy_no_role_or_no_corpus():
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    assert policy.visible_artifacts("x", ["a1"], corpus=None) == set()
