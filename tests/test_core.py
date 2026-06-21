import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, List

import pytest

from groundeval.core import (
    ANSWER_SCHEMAS,
    AccessPolicy,
    AgentTrajectory,
    CorpusAdapter,
    CausalJoinSpec,
    CausalLinkSpec,
    EvalQuestion,
    GatedRuntime,
    LogEvent,
    PerspectiveConfig,
    SearchSpaceSelector,
    SilencePairSpec,
    _get_nested,
    load_events,
)


class DummyCorpus:
    def __init__(self, docs: dict):
        self.docs = docs

    def fetch(self, artifact_id: str, as_of: Optional[str] = None) -> Optional[dict]:
        doc = self.docs.get(artifact_id)
        if doc is None:
            return None
        if as_of and doc.get("timestamp", "") > as_of:
            return None
        return doc

    def search(self, query, artifact_type=None, as_of=None, limit=10):
        results = []
        for doc in self.docs.values():
            if artifact_type and doc.get("subsystem") != artifact_type:
                continue
            if as_of and doc.get("timestamp", "") > as_of:
                continue
            if query.lower() in json.dumps(doc).lower():
                results.append(doc)
            if len(results) >= limit:
                break
        return results

    def timestamp_of(self, artifact_id: str) -> Optional[str]:
        return self.docs.get(artifact_id, {}).get("timestamp")

    def subsystem_of(self, artifact_id: str) -> Optional[str]:
        return self.docs.get(artifact_id, {}).get("subsystem")

    def list_ids(self, subsystem: Optional[str] = None) -> List[str]:
        if not subsystem:
            return list(self.docs.keys())
        return [k for k, v in self.docs.items() if v.get("subsystem") == subsystem]


class DummyPolicy:
    def subsystems_for_role(self, role: str) -> Set[str]:
        return {"jira", "slack"}

    def role_for_actor(self, actor_id: str) -> Optional[str]:
        return "engineer" if actor_id == "alice" else None

    def visible_artifacts(
        self,
        actor_id: str,
        all_artifact_ids: List[str],
        as_of: Optional[str] = None,
        corpus: Optional[CorpusAdapter] = None,
    ) -> Set[str]:
        return {"email-001", "jira-42"}


def test_log_event_from_dict_preserves_extras():
    d = {
        "id": "evt-1",
        "type": "incident",
        "timestamp": "2026-01-15T09:30:00",
        "actors": ["a"],
        "artifact_ids": {"jira": "J-1"},
        "facts": {"sev": "high"},
        "extra_field": 123,
    }
    evt = LogEvent.from_dict(d)
    assert evt.extras == {"extra_field": 123}
    assert evt.resolve("facts.sev") == "high"


def test_search_space_selector_render():
    evt = LogEvent(
        id="e1",
        type="trigger",
        timestamp="2026-01-01T00:00:00",
        artifact_ids={"jira": "J-1"},
    )
    sel = SearchSpaceSelector(subsystem="jira", id_template="{artifact_ids.jira}")
    assert sel.render(evt) == {"mode": "id", "value": "J-1"}

    sel2 = SearchSpaceSelector(
        subsystem="confluence", query_template="postmortem {artifact_ids.jira}"
    )
    assert sel2.render(evt) == {"mode": "query", "value": "postmortem J-1"}


def test_perspective_config_defaults():
    cfg = PerspectiveConfig.from_dict({})
    assert cfg.positive_ratio == 0.5
    assert cfg.require_cross_subsystem_cases is True


def test_causal_link_spec_from_dict_joins():
    raw = {
        "name": "link",
        "cause_event_type": "a",
        "effect_event_type": "b",
        "premise_template": "p",
        "outcome_template": "o",
        "join": [{"cause": "artifact_ids.jira", "effect": "artifact_ids.jira"}],
    }
    spec = CausalLinkSpec.from_dict(raw)
    assert len(spec.join) == 1
    assert spec.join[0].cause == "artifact_ids.jira"


def test_silence_pair_spec_from_dict():
    raw = {
        "trigger_event_type": "t",
        "response_event_type": "r",
        "search_space": [{"subsystem": "jira", "id_template": "{artifact_ids.jira}"}],
    }
    spec = SilencePairSpec.from_dict(raw)
    assert spec.search_space_selectors[0].subsystem == "jira"


def test_gated_runtime_perspective_fetch_gate(tmp_path: Path):
    corpus = DummyCorpus({
        "email-001": {
            "id": "email-001",
            "timestamp": "2026-01-15T09:00:00",
            "subsystem": "email",
        },
        "jira-42": {
            "id": "jira-42",
            "timestamp": "2026-01-15T10:00:00",
            "subsystem": "jira",
        },
    })
    policy = DummyPolicy()
    q = EvalQuestion(
        question_id="q1",
        question_type="PERSPECTIVE",
        question_text="?",
        difficulty="easy",
        ground_truth={},
        actor="alice",
        actor_role="engineer",
        as_of_time="2026-01-15T12:00:00",
        actor_visible_artifacts=["email-001"],
        actor_subsystem_access=["email"],
    )
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        question=q,
        actor_visible_artifacts={"email-001"},
        actor_subsystem_access={"email"},
    )
    assert runtime.fetch("email-001") is not None
    assert runtime.fetch("jira-42") is not None
    traj = runtime.trajectory()
    assert any(c.actor_gate_violation for c in traj.tool_calls)
    assert any(c.subsystem_violation for c in traj.tool_calls)


def test_gated_runtime_silence_no_temporal_gate():
    corpus = DummyCorpus({
        "future-doc": {
            "id": "future-doc",
            "timestamp": "2026-12-01T00:00:00",
            "subsystem": "slack",
        }
    })
    q = EvalQuestion(
        question_id="q1",
        question_type="SILENCE",
        question_text="?",
        difficulty="easy",
        ground_truth={},
    )
    runtime = GatedRuntime(corpus=corpus, policy=DummyPolicy(), question=q)
    assert runtime.fetch("future-doc") is not None


def test_gated_runtime_search_records_calls():
    corpus = DummyCorpus({
        "a1": {"id": "a1", "timestamp": "2026-01-01T00:00:00", "text": "hello"},
    })
    q = EvalQuestion(
        question_id="q1",
        question_type="COUNTERFACTUAL",
        question_text="?",
        difficulty="easy",
        ground_truth={},
        as_of_time="2026-06-01T00:00:00",
    )
    runtime = GatedRuntime(corpus=corpus, policy=DummyPolicy(), question=q)
    results = runtime.search("hello")
    assert len(results) == 1
    assert len(runtime.call_log) == 1
    assert runtime.call_log[0].tool_name == "search_artifacts"


def test_get_nested_dict_access():
    assert _get_nested({"a": {"b": 1}}, "a.b") == 1
    assert _get_nested({"a": {}}, "a.missing") is None
    assert _get_nested({"a": [1, 2]}, "a.0") == 1


def test_load_events_jsonl(tmp_path: Path):
    f = tmp_path / "events.jsonl"
    f.write_text(
        json.dumps({
            "id": "e1",
            "type": "t",
            "timestamp": "2026-01-01T00:00:00",
            "actors": [],
            "artifact_ids": {},
            "facts": {},
        })
        + "\n\n"
        + json.dumps({
            "id": "e2",
            "type": "t",
            "timestamp": "2026-01-02T00:00:00",
            "actors": [],
            "artifact_ids": {},
            "facts": {},
        })
        + "\n"
    )
    events = load_events(f)
    assert len(events) == 2
    assert events[0].id == "e1"


def test_answer_schemas_have_required_keys():
    assert "PERSPECTIVE" in ANSWER_SCHEMAS
    assert "COUNTERFACTUAL" in ANSWER_SCHEMAS
    assert "SILENCE" in ANSWER_SCHEMAS


def test_gated_runtime_exact_timestamp_boundary():
    """ts == as_of must NOT be gated (boundary is inclusive)."""

    class MinCorp:
        def fetch(self, aid, as_of=None):
            return {"id": "a1", "timestamp": "2026-01-15T09:00:00"}

        def timestamp_of(self, aid):
            return "2026-01-15T09:00:00"

        def subsystem_of(self, aid):
            return "jira"

        def list_ids(self, subsystem=None):
            return ["a1"]

        def search(self, *a, **k):
            return []

    class MinPol:
        def subsystems_for_role(self, role):
            return {"jira"}

        def role_for_actor(self, actor):
            return "eng"

        def visible_artifacts(self, *a, **k):
            return {"a1"}

    q = EvalQuestion(
        question_id="q1",
        question_type="COUNTERFACTUAL",
        question_text="?",
        difficulty="easy",
        ground_truth={},
        as_of_time="2026-01-15T09:00:00",
    )
    rt = GatedRuntime(corpus=MinCorp(), policy=MinPol(), question=q)
    assert rt.fetch("a1") is not None


def test_gated_runtime_trajectory_no_calls():
    rt = GatedRuntime(
        corpus=DummyCorpus({}),
        policy=DummyPolicy(),
        question=EvalQuestion(
            question_id="q1",
            question_type="PERSPECTIVE",
            question_text="?",
            difficulty="easy",
            ground_truth={},
        ),
    )
    traj = rt.trajectory()
    assert traj.tool_calls == []
    assert traj.dead_ends_hit == 0
    assert traj.dead_ends_recovered == 0


def test_get_nested_invalid_paths():
    assert _get_nested({"a": 1}, "a.b") is None
    assert _get_nested({"a": [1]}, "a.foo") is None
    assert _get_nested({"a": [1]}, "a.5") is None
    assert _get_nested(None, "a") is None


def test_load_events_empty_and_blanks(tmp_path: Path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    assert load_events(p) == []

    p2 = tmp_path / "blanks.jsonl"
    p2.write_text("\n\n  \n")
    assert load_events(p2) == []


def test_eval_question_defaults():
    q = EvalQuestion(
        question_id="q1",
        question_type="SILENCE",
        question_text="?",
        difficulty="easy",
        ground_truth={},
    )
    assert q.actor is None
    assert q.actor_visible_artifacts is None
    assert q.cross_subsystem is False


def test_agent_trajectory_defaults():
    t = AgentTrajectory(question_id="q1", question_type="SILENCE")
    assert t.tool_calls == []
    assert t.total_latency_ms == 0.0
    assert t.budget_exceeded is False
