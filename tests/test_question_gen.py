from datetime import timedelta
from typing import List

import pytest

from groundeval.core import (
    CausalLinkSpec,
    LogEvent,
    SilencePairSpec,
    PerspectiveConfig,
    CausalJoinSpec,
)
from groundeval.adapters import YamlAccessPolicy
from groundeval.question_gen import (
    CausalLinkIndexer,
    AbsenceCatalogBuilder,
    QuestionGenerator,
)


def make_event(
    eid: str,
    etype: str,
    ts: str,
    actors: List[str],
    artifact_ids: dict,
    facts: dict = None,
):
    return LogEvent(
        id=eid,
        type=etype,
        timestamp=ts,
        actors=actors,
        artifact_ids=artifact_ids,
        facts=facts or {},
    )


def test_link_indexer_finds_cause_effect_pair():
    events = [
        make_event(
            "c1", "escalation_opened", "2026-01-01T00:00:00", ["a"], {"jira": "J-1"}
        ),
        make_event(
            "e1", "postmortem_created", "2026-01-02T00:00:00", ["a"], {"jira": "J-1"}
        ),
    ]
    spec = CausalLinkSpec(
        name="esc_to_post",
        cause_event_type="escalation_opened",
        effect_event_type="postmortem_created",
        premise_template="p",
        outcome_template="o",
        max_gap_days=7,
    )
    indexer = CausalLinkIndexer(events, [spec])
    links = indexer.build()
    assert len(links) == 1
    assert links[0].cause_event_id == "c1"
    assert links[0].effect_event_id == "e1"


def test_link_indexer_respects_max_gap():
    events = [
        make_event(
            "c1", "escalation_opened", "2026-01-01T00:00:00", ["a"], {"jira": "J-1"}
        ),
        make_event(
            "e1", "postmortem_created", "2026-01-30T00:00:00", ["a"], {"jira": "J-1"}
        ),
    ]
    spec = CausalLinkSpec(
        name="esc_to_post",
        cause_event_type="escalation_opened",
        effect_event_type="postmortem_created",
        premise_template="p",
        outcome_template="o",
        max_gap_days=7,
    )
    indexer = CausalLinkIndexer(events, [spec])
    links = indexer.build()
    assert len(links) == 0


def test_link_indexer_join_conditions():
    events = [
        make_event(
            "c1", "escalation_opened", "2026-01-01T00:00:00", ["a"], {"jira": "J-1"}
        ),
        make_event(
            "e1", "postmortem_created", "2026-01-02T00:00:00", ["a"], {"jira": "J-1"}
        ),
        make_event(
            "e2", "postmortem_created", "2026-01-02T00:00:00", ["a"], {"jira": "J-2"}
        ),
    ]
    spec = CausalLinkSpec(
        name="esc_to_post",
        cause_event_type="escalation_opened",
        effect_event_type="postmortem_created",
        premise_template="p",
        outcome_template="o",
        max_gap_days=7,
        join=[CausalJoinSpec(cause="artifact_ids.jira", effect="artifact_ids.jira")],
    )
    indexer = CausalLinkIndexer(events, [spec])
    links = indexer.build()
    assert len(links) == 1
    assert links[0].effect_event_id == "e1"


def test_absence_builder_finds_missing_response():
    events = [
        make_event(
            "t1", "escalation_opened", "2026-01-01T00:00:00", ["a"], {"jira": "J-1"}
        ),
    ]
    spec = SilencePairSpec(
        trigger_event_type="escalation_opened",
        response_event_type="postmortem_created",
        max_gap_days=7,
    )
    builder = AbsenceCatalogBuilder(events, [spec])
    absences, confirmed = builder.build()
    assert len(absences) == 1
    assert len(confirmed) == 0
    assert absences[0].trigger_event_id == "t1"


def test_absence_builder_confirms_present_response():
    events = [
        make_event(
            "t1", "escalation_opened", "2026-01-01T00:00:00", ["a"], {"jira": "J-1"}
        ),
        make_event(
            "r1", "postmortem_created", "2026-01-02T00:00:00", ["a"], {"jira": "J-1"}
        ),
    ]
    spec = SilencePairSpec(
        trigger_event_type="escalation_opened",
        response_event_type="postmortem_created",
        max_gap_days=7,
    )
    builder = AbsenceCatalogBuilder(events, [spec])
    absences, confirmed = builder.build()
    assert len(absences) == 0
    assert len(confirmed) == 1


def test_question_generator_produces_all_three_tracks():
    events = [
        make_event(
            "e1", "incident_opened", "2026-01-01T00:00:00", ["alice"], {"jira": "J-1"}
        ),
        make_event(
            "e2", "note_added", "2026-01-02T00:00:00", ["alice"], {"jira": "J-1"}
        ),
        make_event(
            "e3", "ticket_closed", "2026-01-03T00:00:00", ["alice"], {"jira": "J-1"}
        ),
    ]
    policy = YamlAccessPolicy({
        "actors": {"alice": "engineer"},
        "roles": {"engineer": {"subsystems": ["jira"]}},
    })

    link_spec = CausalLinkSpec(
        name="incident_note",
        cause_event_type="incident_opened",
        effect_event_type="note_added",
        premise_template="if incident p",
        outcome_template="note o",
        max_gap_days=7,
        join=[CausalJoinSpec(cause="artifact_ids.jira", effect="artifact_ids.jira")],
    )
    indexer = CausalLinkIndexer(events, [link_spec])
    links = indexer.build()

    silence_spec = SilencePairSpec(
        trigger_event_type="incident_opened",
        response_event_type="ticket_closed",
        max_gap_days=7,
    )
    builder = AbsenceCatalogBuilder(events, [silence_spec])
    absences, confirmed = builder.build()

    class MiniCorpus:
        def list_ids(self, subsystem=None):
            return ["J-1"]

        def subsystem_of(self, aid):
            return "jira"

    gen = QuestionGenerator(
        events=events,
        causal_links=links,
        absence_records=absences,
        confirmed_records=confirmed,
        policy=policy,
        corpus=MiniCorpus(),
        perspective_config=PerspectiveConfig(),
    )
    questions = gen.generate()
    types = {q.question_type for q in questions}
    assert "PERSPECTIVE" in types
    assert "COUNTERFACTUAL" in types
    assert len(questions) > 0


def test_render_template_substitution():
    from groundeval.question_gen import CausalLinkIndexer

    indexer = CausalLinkIndexer([], [])
    cause = make_event(
        "c1", "t", "2026-01-01T00:00:00", ["a"], {"jira": "J-1"}, {"sev": "high"}
    )
    effect = make_event(
        "e1", "t", "2026-01-02T00:00:00", ["b"], {"jira": "J-1"}, {"note": "done"}
    )
    out = indexer._render(
        "Severity {cause.facts.sev} note {effect.facts.note}", cause, effect
    )
    assert out == "Severity high note done"


def test_link_indexer_no_matching_events():
    events = [
        make_event("c1", "wrong_type", "2026-01-01T00:00:00", ["a"], {"jira": "J-1"}),
    ]
    spec = CausalLinkSpec(
        name="x",
        cause_event_type="escalation_opened",
        effect_event_type="postmortem_created",
        premise_template="p",
        outcome_template="o",
    )
    indexer = CausalLinkIndexer(events, [spec])
    assert indexer.build() == []


def test_absence_builder_exact_boundary_gap():
    """Response exactly at max_gap_days boundary should count as present."""
    trigger = make_event("t1", "esc", "2026-01-01T00:00:00", ["a"], {"jira": "J-1"})
    response = make_event("r1", "post", "2026-01-03T00:00:00", ["a"], {"jira": "J-1"})
    spec = SilencePairSpec(
        trigger_event_type="esc",
        response_event_type="post",
        max_gap_days=2,
    )
    builder = AbsenceCatalogBuilder([trigger, response], [spec])
    absences, confirmed = builder.build()
    assert len(absences) == 0
    assert len(confirmed) == 1


def test_render_missing_placeholder_blank():
    from groundeval.question_gen import CausalLinkIndexer

    indexer = CausalLinkIndexer([], [])
    cause = make_event("c1", "t", "2026-01-01T00:00:00", ["a"], {"jira": "J-1"})
    effect = make_event("e1", "t", "2026-01-02T00:00:00", ["b"], {"jira": "J-1"})
    out = indexer._render("Hello {cause.missing}", cause, effect)
    assert out == "Hello "


def test_question_generator_empty_inputs():
    class DummyPol:
        def subsystems_for_role(self, role):
            return set()

        def role_for_actor(self, actor):
            return None

        def visible_artifacts(self, *a, **k):
            return set()

    class DummyCorp:
        def list_ids(self, subsystem=None):
            return []

        def subsystem_of(self, aid):
            return None

    gen = QuestionGenerator(
        events=[],
        causal_links=[],
        absence_records=[],
        confirmed_records=[],
        policy=DummyPol(),
        corpus=DummyCorp(),
    )
    assert gen.generate() == []


def test_join_matches_none_values():
    from groundeval.question_gen import CausalLinkIndexer

    indexer = CausalLinkIndexer([], [])
    cause = make_event("c1", "t", "2026-01-01T00:00:00", ["a"], {})
    effect = make_event("e1", "t", "2026-01-02T00:00:00", ["b"], {})
    assert (
        indexer._join_matches(
            cause,
            effect,
            [CausalJoinSpec(cause="artifact_ids.jira", effect="artifact_ids.jira")],
        )
        is False
    )
