"""
groundeval/question_gen.py
================================
Generates PERSPECTIVE, COUNTERFACTUAL, and SILENCE questions
from a generic event log + user-supplied causal link / silence pair specs.

No domain-specific logic. Domain coupling is eliminated by treating
causal links and silence pairs as configuration rather than hardcoded dispatch.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRACK 1 — PERSPECTIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Questions scoped to what a specific actor could have known at a specific moment,
given their actual subsystem access and information horizon. Balanced across
positive (could know), negative-permission (blocked subsystem), and
negative-temporal (event hadn't happened yet) cases.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRACK 2 — COUNTERFACTUAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Questions of the form "if X had been different, would Y have occurred?"
Only generated from explicit CausalLink records — no inference. Ground truth
is always derivable from the link without reasoning over the corpus.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRACK 3 — SILENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Questions about things that did NOT happen. The event log is the arbiter:
if no response event fired, absence is ground truth. Each question includes
an expected_search_space — the artifact IDs the agent MUST check before
concluding absence. Correct "no" without searching scores 0 on trajectory.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Design principles
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Ground truth is always derived from the event log. LLMs only write prose.
- Question prose generation includes a structured validation loop (3 attempts).
- Actor visibility cones are first-class data structures, not a scoring afterthought.
- Subsystem access is explicitly modeled per actor via AccessPolicy.
- The absence catalog is built by pattern-matching expected event pairs, not heuristics.
- SILENCE sampling is stratified by trigger_event_type to prevent clustering.
- COUNTERFACTUAL questions rotate across 5 phrasing styles to prevent monotony.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from .core import (
    AbsenceRecord,
    CausalJoinSpec,
    CausalLink,
    CausalLinkSpec,
    EvalQuestion,
    LogEvent,
    PerspectiveConfig,
    SilencePairSpec,
    _get_nested,
    ANSWER_SCHEMAS,
)
from .core import AccessPolicy, CorpusAdapter

logger = logging.getLogger("groundeval.question_gen")


_COUNTERFACTUAL_STYLES: List[str] = [
    "Phrase it as a direct counterfactual: 'If X had not happened, would Y have occurred?'",
    "Phrase it as a hypothetical: 'Had X not taken place, would Y still have happened?'",
    "Phrase it from the outcome perspective: 'Would Y have occurred without X happening first?'",
    "Phrase it as an investigative question: 'Was Y a direct consequence of X, or would it have happened regardless?'",
    "Phrase it as a dependency question: 'Did Y depend on X occurring, or was it independently triggered?'",
]

_NEGATIVE_COUNTERFACTUAL_STYLES: List[str] = [
    "Phrase it to suggest the cause was necessary: 'If X had not happened, would Y still have occurred?'",
    "Phrase it as a dependency question: 'Was Y dependent on X, or would it have happened regardless?'",
    "Phrase it from the effect perspective: 'Would Y have been prevented if only X had been different?'",
    "Phrase it as a challenge: 'Can we be sure X was actually required for Y, or would Y have happened anyway?'",
    "Phrase it as an alternate history: 'Had X not taken place, would the outcome still have been Y?'",
]

_SILENCE_STYLES: List[str] = [
    "Phrase it as a process compliance question a manager would ask.",
    "Phrase it as an audit question checking whether a response was documented.",
    "Phrase it as an operational question about whether follow-up occurred.",
    "Phrase it as a gap analysis question about whether proper procedure was followed.",
]

_PERSPECTIVE_STYLES: List[str] = [
    "Phrase it from the actor's point of view, as if an auditor is checking their awareness.",
    "Phrase it as a knowledge audit: did this actor have access to the facts at the time?",
    "Phrase it as an accountability question about information silos.",
    "Phrase it as an investigation into whether the actor was in the loop.",
    "Phrase it neutrally: simply ask what the actor could have known given their access.",
]


_TRIVIAL_SEARCH_SPACE: Set[str] = set()

_MAX_QUESTIONS_PER_ACTOR = 5
_MAX_QUESTIONS_PER_EVENT_TYPE = 5


class CausalLinkIndexer:
    """
    Scans the event log for cause->effect pairs matching user-supplied specs.

    For each CausalLinkSpec:
      1. Find all events of cause_event_type (optionally filtered by match_field/value)
      2. For each cause event, find the chronologically next event of effect_event_type
         within max_gap_days (if specified) that also satisfies the join conditions
      3. Emit a CausalLink record with templated premise/outcome strings

    No domain knowledge required — the user's spec IS the domain knowledge.
    """

    MAX_LINKS_PER_SPEC = 5

    def __init__(self, events: List[LogEvent], specs: List[CausalLinkSpec]):
        self._events = sorted(events, key=lambda e: e.timestamp)
        self._specs = specs

    def build(self) -> List[CausalLink]:
        links: List[CausalLink] = []
        for spec in self._specs:
            links.extend(self._index_spec(spec))
        return links

    def _index_spec(self, spec: CausalLinkSpec) -> List[CausalLink]:
        cause_candidates = [
            e
            for e in self._events
            if e.type == spec.cause_event_type
            and self._match_filter(e, spec.match_field, spec.match_value)
        ]
        random.shuffle(cause_candidates)

        links = []
        for cause in cause_candidates:
            if len(links) >= self.MAX_LINKS_PER_SPEC:
                break
            effect = self._find_effect(cause, spec)
            if effect is None:
                continue

            cause_artifacts = self._collect_artifacts(cause)
            effect_artifacts = self._collect_artifacts(effect)
            all_artifacts = cause_artifacts + effect_artifacts

            links.append(
                CausalLink(
                    link_type=spec.name,
                    cause_event_id=cause.id,
                    cause_event_type=cause.type,
                    cause_timestamp=cause.timestamp,
                    effect_event_id=effect.id,
                    effect_event_type=effect.type,
                    effect_timestamp=effect.timestamp,
                    actors=list(set(cause.actors + effect.actors)),
                    counterfactual_premise=self._render(
                        spec.premise_template, cause, effect
                    ),
                    counterfactual_outcome=self._render(
                        spec.outcome_template, cause, effect
                    ),
                    outcome_changed=spec.outcome_changed,
                    cause_artifact_ids=list(set(cause_artifacts)),
                    effect_artifact_ids=list(set(effect_artifacts)),
                    evidence_artifact_ids=list(set(all_artifacts)),
                    mechanism_aliases=spec.mechanism_aliases,
                )
            )

        return links

    def _find_effect(self, cause: LogEvent, spec: CausalLinkSpec) -> Optional[LogEvent]:
        """
        Find the first effect event after cause, within max_gap_days,
        that also satisfies all join conditions.
        """
        for event in self._events:
            if event.timestamp <= cause.timestamp:
                continue
            if event.type != spec.effect_event_type:
                continue
            if spec.max_gap_days is not None:
                gap = (
                    datetime.fromisoformat(event.timestamp)
                    - datetime.fromisoformat(cause.timestamp)
                ).total_seconds() / 86400
                if gap > spec.max_gap_days:
                    break
            if not self._join_matches(cause, event, spec.join):
                continue
            return event
        return None

    def _join_matches(
        self, cause: LogEvent, effect: LogEvent, joins: List[CausalJoinSpec]
    ) -> bool:
        """Evaluate all join conditions between cause and effect events."""
        if not joins:
            return True
        for j in joins:
            cause_val = cause.resolve(j.cause)
            effect_val = effect.resolve(j.effect)
            if cause_val is None or effect_val is None:
                return False
            c_vals = set(cause_val) if isinstance(cause_val, list) else {cause_val}
            e_vals = set(effect_val) if isinstance(effect_val, list) else {effect_val}
            if not (c_vals & e_vals):
                return False
        return True

    def _match_filter(
        self,
        event: LogEvent,
        match_field: Optional[str],
        match_value: Any,
    ) -> bool:
        if match_field is None:
            return True
        val = event.resolve(match_field)
        return val == match_value

    def _collect_artifacts(self, event: LogEvent) -> List[str]:
        ids = []
        for val in event.artifact_ids.values():
            if isinstance(val, list):
                ids.extend(str(v) for v in val if v)
            elif val:
                ids.append(str(val))
        return ids

    def _render(self, template: str, cause: LogEvent, effect: LogEvent) -> str:
        """
        Simple template substitution.
        Supported placeholders: {cause.type}, {effect.type},
        {cause.actors[0]}, {cause.facts.KEY}, {effect.facts.KEY}
        """

        def replacer(m):
            path = m.group(1)
            if path.startswith("cause."):
                return str(_get_nested(cause.to_dict(), path[len("cause.") :]) or "")
            if path.startswith("effect."):
                return str(_get_nested(effect.to_dict(), path[len("effect.") :]) or "")
            return m.group(0)

        return re.sub(r"\{([^}]+)\}", replacer, template)


class AbsenceCatalogBuilder:
    """
    Finds trigger events with no matching response in the event log.

    For each SilencePairSpec:
      1. Find all trigger events of trigger_event_type
      2. For each trigger, check whether a response_event_type followed
         (within max_gap_days)
      3. If not -> AbsenceRecord (verified absence, not an inference)
    """

    def __init__(
        self,
        events: List[LogEvent],
        specs: List[SilencePairSpec],
        corpus: Optional[CorpusAdapter] = None,
    ):
        self._events = sorted(events, key=lambda e: e.timestamp)
        self._specs = specs
        self._corpus = corpus

    def build(self) -> Tuple[List[AbsenceRecord], List[LogEvent]]:
        absences: List[AbsenceRecord] = []
        confirmed: List[LogEvent] = []

        for spec in self._specs:
            a, c = self._scan_spec(spec)
            absences.extend(a)
            confirmed.extend(c)

        return absences, confirmed

    def _scan_spec(
        self, spec: SilencePairSpec
    ) -> Tuple[List[AbsenceRecord], List[LogEvent]]:
        triggers = [
            e
            for e in self._events
            if e.type == spec.trigger_event_type
            and self._match_filter(e, spec.match_field, spec.match_value)
        ]

        absences: List[AbsenceRecord] = []
        confirmed: List[LogEvent] = []

        for trigger in triggers:
            response = self._find_response(trigger, spec)
            if response is None:
                search_space = self._build_search_space(trigger, spec)
                absences.append(
                    AbsenceRecord(
                        trigger_event_id=trigger.id,
                        trigger_event_type=trigger.type,
                        trigger_timestamp=trigger.timestamp,
                        trigger_actors=list(trigger.actors),
                        expected_response_type=spec.response_event_type,
                        expected_search_space=search_space,
                        subsystem=spec.search_space_subsystems[0]
                        if spec.search_space_subsystems
                        else "",
                    )
                )
            else:
                confirmed.append(response)

        return absences, confirmed

    def _find_response(
        self, trigger: LogEvent, spec: SilencePairSpec
    ) -> Optional[LogEvent]:
        for event in self._events:
            if event.timestamp <= trigger.timestamp:
                continue
            if event.type != spec.response_event_type:
                continue
            if spec.max_gap_days is not None:
                gap = (
                    datetime.fromisoformat(event.timestamp)
                    - datetime.fromisoformat(trigger.timestamp)
                ).total_seconds() / 86400
                if gap > spec.max_gap_days:
                    break
            if not self._join_matches(trigger, event, spec.join):
                continue
            return event
        return None

    def _join_matches(
        self, trigger: LogEvent, response: LogEvent, joins: List[CausalJoinSpec]
    ) -> bool:
        if not joins:
            return True
        for j in joins:
            trigger_val = trigger.resolve(j.cause)
            response_val = response.resolve(j.effect)
            if trigger_val is None or response_val is None:
                return False
            c_vals = (
                set(trigger_val) if isinstance(trigger_val, list) else {trigger_val}
            )
            e_vals = (
                set(response_val) if isinstance(response_val, list) else {response_val}
            )
            if not (c_vals & e_vals):
                return False
        return True

    def _build_search_space(
        self, trigger: LogEvent, spec: SilencePairSpec
    ) -> List[str]:
        """
        Return artifact IDs the agent must search to verify absence.
        Combines artifacts from the trigger event with any corpus IDs
        in the relevant subsystems.

        If search_space_selectors are declared in the spec, they take
        precedence and produce precise template-rendered entries.
        """
        space: Set[str] = set()

        for val in trigger.artifact_ids.values():
            if isinstance(val, list):
                space.update(str(v) for v in val if v)
            elif val:
                space.add(str(val))

        if spec.search_space_selectors:
            for selector in spec.search_space_selectors:
                rendered = selector.render(trigger)
                if rendered["mode"] == "id":
                    space.add(rendered["value"])
                else:
                    if self._corpus:
                        try:
                            hits = self._corpus.search(
                                rendered["value"],
                                artifact_type=selector.subsystem,
                                limit=5,
                            )
                            for h in hits:
                                hid = str(h.get("id", h.get("_id", "")))
                                if hid:
                                    space.add(hid)
                        except Exception:
                            space.add(f"[{selector.subsystem}] {rendered['value']}")
                    else:
                        space.add(f"[{selector.subsystem}] {rendered['value']}")
            if spec.search_space_subsystems and self._corpus:
                for subsystem in spec.search_space_subsystems:
                    space.update(self._corpus.list_ids(subsystem=subsystem)[:5])
        elif self._corpus:
            for subsystem in spec.search_space_subsystems:
                space.update(self._corpus.list_ids(subsystem=subsystem)[:20])

        return sorted(space)

    def _match_filter(
        self, event: LogEvent, match_field: Optional[str], match_value: Any
    ) -> bool:
        if match_field is None:
            return True
        val = event.resolve(match_field)
        return val == match_value


class QuestionGenerator:
    """
    Produces EvalQuestion records for all three tracks.

    For question prose: either use an LLM (if llm_fn is provided) or
    fall back to deterministic templates. The ground_truth is always
    derived deterministically — the LLM only writes the question surface.

    llm_fn signature: (prompt: str) -> str

    trivial_search_space: set of artifact ID strings considered too generic
    to count as meaningful search evidence for SILENCE questions. Defaults to
    the module-level _TRIVIAL_SEARCH_SPACE set (empty by default). Pass a
    domain-specific set such as {"jira", "confluence", "slack"} to skip
    SILENCE questions whose expected_search_space contains only those entries.
    """

    MAX_PERSPECTIVE = 40
    MAX_COUNTERFACTUAL = 40
    MAX_SILENCE = 40

    def __init__(
        self,
        events: List[LogEvent],
        causal_links: List[CausalLink],
        absence_records: List[AbsenceRecord],
        confirmed_records: List[LogEvent],
        policy: AccessPolicy,
        corpus: Optional[CorpusAdapter] = None,
        llm_fn=None,
        perspective_actors: Optional[List[str]] = None,
        perspective_config: Optional[PerspectiveConfig] = None,
        trivial_search_space: Optional[Set[str]] = None,
    ):
        self._events = sorted(events, key=lambda e: e.timestamp)
        self._links = causal_links
        self._absences = absence_records
        self._confirmed = confirmed_records
        self._policy = policy
        self._corpus = corpus
        self._llm_fn = llm_fn
        self._perspective_actors = perspective_actors
        self._perspective_config = perspective_config or PerspectiveConfig()
        self._trivial_search_space = (
            trivial_search_space
            if trivial_search_space is not None
            else _TRIVIAL_SEARCH_SPACE
        )

    def generate(self) -> List[EvalQuestion]:
        questions: List[EvalQuestion] = []
        questions.extend(self._perspective_questions())
        questions.extend(self._counterfactual_questions())
        questions.extend(self._silence_questions())
        random.shuffle(questions)
        return questions

    def _perspective_questions(self) -> List[EvalQuestion]:
        """
        Generate perspective questions with explicit balance across:
          - Positive: actor COULD have known
          - Negative (permission): actor could NOT know due to subsystem boundary
          - Negative (temporal): actor could NOT know because target happened after as_of

        Enforces per-actor and per-event-type caps so the eval set reflects
        organizational breadth rather than one dominant actor or event type.
        """
        questions: List[EvalQuestion] = []
        actors = self._perspective_actors or self._infer_actors()
        all_artifact_ids = self._corpus.list_ids() if self._corpus else []

        positive: List[EvalQuestion] = []
        neg_permission: List[EvalQuestion] = []
        neg_temporal: List[EvalQuestion] = []

        # Track quotas
        actor_counts: Dict[str, int] = defaultdict(int)
        event_type_counts: Dict[str, int] = defaultdict(int)

        for actor in actors:
            role = self._policy.role_for_actor(actor)
            if not role:
                continue
            if actor_counts[actor] >= _MAX_QUESTIONS_PER_ACTOR:
                continue
            accessible = self._policy.subsystems_for_role(role)

            pos, perm, tmp = self._build_perspective_for_actor(
                actor,
                role,
                accessible,
                all_artifact_ids,
                actor_counts=actor_counts,
                event_type_counts=event_type_counts,
            )
            positive.extend(pos)
            neg_permission.extend(perm)
            neg_temporal.extend(tmp)

        cfg = self._perspective_config
        total_target = self.MAX_PERSPECTIVE
        n_pos = min(len(positive), int(total_target * cfg.positive_ratio))
        n_perm = min(
            len(neg_permission), int(total_target * cfg.negative_permission_ratio)
        )
        n_tmp = min(len(neg_temporal), int(total_target * cfg.negative_temporal_ratio))

        shortfall = total_target - (n_pos + n_perm + n_tmp)
        if shortfall > 0:
            for bucket, count_ref in [
                (positive, "n_pos"),
                (neg_permission, "n_perm"),
                (neg_temporal, "n_tmp"),
            ]:
                extra = min(shortfall, len(bucket) - locals()[count_ref])
                locals()[count_ref] += extra
                shortfall -= extra

        questions.extend(
            random.sample(positive, min(n_pos, len(positive))) if positive else []
        )
        questions.extend(
            random.sample(neg_permission, min(n_perm, len(neg_permission)))
            if neg_permission
            else []
        )
        questions.extend(
            random.sample(neg_temporal, min(n_tmp, len(neg_temporal)))
            if neg_temporal
            else []
        )

        if cfg.require_cross_subsystem_cases:
            has_cross = any(
                q.cross_subsystem
                for q in questions
                if q.question_type == "PERSPECTIVE"
                and not q.ground_truth.get("could_actor_have_known")
            )
            if not has_cross and neg_permission:
                cross_candidates = [q for q in neg_permission if q.cross_subsystem]
                if cross_candidates:
                    for i, q in enumerate(questions):
                        if q.question_type == "PERSPECTIVE" and not q.ground_truth.get(
                            "could_actor_have_known"
                        ):
                            questions[i] = random.choice(cross_candidates)
                            break

        return questions[:total_target]

    def _build_perspective_for_actor(
        self,
        actor: str,
        role: str,
        accessible: Set[str],
        all_artifact_ids: List[str],
        actor_counts: Dict[str, int],
        event_type_counts: Dict[str, int],
    ) -> Tuple[List[EvalQuestion], List[EvalQuestion], List[EvalQuestion]]:
        """
        Returns (positive_qs, neg_permission_qs, neg_temporal_qs).
        """
        positive: List[EvalQuestion] = []
        neg_permission: List[EvalQuestion] = []
        neg_temporal: List[EvalQuestion] = []

        actor_events = [e for e in self._events if actor in e.actors]
        if not actor_events:
            return positive, neg_permission, neg_temporal

        candidate_events = random.sample(actor_events, min(4, len(actor_events)))

        for pivot_event in candidate_events:
            as_of = pivot_event.timestamp

            visible = self._policy.visible_artifacts(
                actor_id=actor,
                all_artifact_ids=all_artifact_ids,
                as_of=as_of,
                corpus=self._corpus,
            )

        for target_event in self._events:
            if target_event.id == pivot_event.id:
                continue
            if not target_event.artifact_ids:
                continue

            # Enforce per-event-type cap
            if event_type_counts[target_event.type] >= _MAX_QUESTIONS_PER_EVENT_TYPE:
                continue

            target_subsystems = set()
            for val in target_event.artifact_ids.values():
                aids = val if isinstance(val, list) else [val]
                for aid in aids:
                    if aid:
                        sub = self._artifact_subsystem(aid)
                        if sub:
                            target_subsystems.add(sub)

            if not target_subsystems:
                continue

            blocked = target_subsystems - accessible
            temporal_blocked = target_event.timestamp > as_of

            if (
                temporal_blocked
                and not blocked
                and self._temporal_gap_reasonable(as_of, target_event.timestamp)
            ):
                q = self._make_perspective_question(
                    actor,
                    role,
                    as_of,
                    target_event,
                    visible,
                    accessible,
                    could_know=False,
                    reason_type="temporal",
                )
                if q:
                    event_type_counts[target_event.type] += 1
                    actor_counts[actor] += 1
                    neg_temporal.append(q)
                continue

            if blocked:
                q = self._make_perspective_question(
                    actor,
                    role,
                    as_of,
                    target_event,
                    visible,
                    accessible,
                    could_know=False,
                    reason_type="permission",
                )
                if q:
                    event_type_counts[target_event.type] += 1
                    actor_counts[actor] += 1
                    neg_permission.append(q)
                continue

            if not blocked and not temporal_blocked:
                q = self._make_perspective_question(
                    actor,
                    role,
                    as_of,
                    target_event,
                    visible,
                    accessible,
                    could_know=True,
                    reason_type="positive",
                )
                if q:
                    event_type_counts[target_event.type] += 1
                    actor_counts[actor] += 1
                    positive.append(q)

            if actor_counts[actor] >= _MAX_QUESTIONS_PER_ACTOR:
                break

                if len(positive) + len(neg_permission) + len(neg_temporal) >= 9:
                    break
            if len(positive) + len(neg_permission) + len(neg_temporal) >= 9:
                break

        return positive, neg_permission, neg_temporal

    def _temporal_gap_reasonable(self, as_of: str, target_ts: str) -> bool:
        try:
            gap = (
                datetime.fromisoformat(target_ts) - datetime.fromisoformat(as_of)
            ).total_seconds() / 86400
            return gap <= 7
        except Exception:
            return False

    def _make_perspective_question(
        self,
        actor: str,
        role: str,
        as_of: str,
        target_event: LogEvent,
        visible: Set[str],
        accessible: Set[str],
        could_know: bool,
        reason_type: str,
    ) -> Optional[EvalQuestion]:
        _blocked: Set[str] = set()
        for val in target_event.artifact_ids.values():
            aids = val if isinstance(val, list) else [val]
            for aid in aids:
                if not aid:
                    continue
                sub = self._artifact_subsystem(aid)
                if sub is not None and sub not in accessible:
                    _blocked.add(sub)
        blocked_subsystems = sorted(_blocked)

        cross_subsystem = len(blocked_subsystems) > 0
        if could_know:
            difficulty = "easy"
        elif reason_type == "temporal":
            difficulty = "medium"
        else:
            difficulty = "hard" if cross_subsystem else "medium"

        target_artifact_ids = set()
        for val in target_event.artifact_ids.values():
            aids = val if isinstance(val, list) else [val]
            for aid in aids:
                if aid and aid in visible:
                    target_artifact_ids.add(aid)

        ground_truth = {
            "could_actor_have_known": could_know,
            "reason": (
                f"Event {target_event.type} involved subsystems "
                f"{blocked_subsystems} not accessible to role '{role}'"
                if not could_know and blocked_subsystems
                else (
                    f"Event {target_event.type} occurred after {as_of[:10]} "
                    f"and was not yet visible"
                    if not could_know and reason_type == "temporal"
                    else f"Actor {actor} had access to all relevant subsystems and the event was within temporal bounds"
                )
            ),
            "blocked_subsystems": blocked_subsystems,
            "evidence_artifacts": sorted(target_artifact_ids),
        }

        question_text = self._prose_perspective(actor, as_of, target_event, could_know)
        if not question_text:
            return None

        qid = self._make_id("P", actor, as_of, target_event.type)
        return EvalQuestion(
            question_id=qid,
            question_type="PERSPECTIVE",
            question_text=question_text,
            difficulty=difficulty,
            ground_truth=ground_truth,
            actor=actor,
            actor_role=role,
            as_of_time=as_of,
            actor_visible_artifacts=sorted(visible),
            actor_subsystem_access=sorted(accessible),
            cross_subsystem=cross_subsystem,
            expected_answer_schema=ANSWER_SCHEMAS["PERSPECTIVE"],
        )

    def _prose_perspective(
        self,
        actor: str,
        as_of: str,
        target_event: LogEvent,
        could_know: bool,
    ) -> Optional[str]:
        event_desc = target_event.type.replace("_", " ")
        actors_str = (
            ", ".join(target_event.actors[:2]) if target_event.actors else "others"
        )

        template = (
            f"Based only on what {actor} had access to as of {as_of[:10]}, "
            f"could they have known about the {event_desc} involving {actors_str}?"
        )

        if not self._llm_fn:
            return template

        style = random.choice(_PERSPECTIVE_STYLES)

        prompt = (
            f"{style}\n\n"
            f"Actor: {actor}.\n"
            f"As-of date: {as_of[:10]}.\n"
            f"Event: a {event_desc} involving {actors_str}.\n\n"
            f"Requirements:\n"
            f"- The question MUST end with a question mark.\n"
            f"- It MUST explicitly reference the actor ({actor}) and the date ({as_of[:10]}).\n"
            f"- Do NOT reveal whether the answer is yes or no.\n"
            f"- Do NOT include artifact IDs or internal references.\n"
            f"- Use natural language. 15-80 words.\n"
            f"Output only the question text."
        )

        for attempt in range(3):
            try:
                result = self._llm_fn(prompt).strip()
                if self._validate_prose(
                    result, ground_truth_str="", question_type="PERSPECTIVE"
                ):
                    return result
                else:
                    logger.debug(
                        f"[question_gen] Perspective prose validation failed "
                        f"(attempt {attempt + 1}): {result[:80]}"
                    )
            except Exception as exc:
                logger.warning(
                    f"[question_gen] Perspective prose generation error "
                    f"(attempt {attempt + 1}): {exc}"
                )

        return template

    def _counterfactual_questions(self) -> List[EvalQuestion]:
        questions: List[EvalQuestion] = []
        all_artifact_ids = self._corpus.list_ids() if self._corpus else []

        effect_cause_map: Dict[str, List[CausalLink]] = defaultdict(list)
        for link in self._links:
            effect_cause_map[link.effect_event_id].append(link)

        positive_candidates = random.sample(
            self._links, min(self.MAX_COUNTERFACTUAL, len(self._links))
        )

        actor_counts: Dict[str, int] = defaultdict(int)
        event_type_counts: Dict[str, int] = defaultdict(int)

        seen_effects: Set[str] = set()
        for link in positive_candidates:
            if link.effect_event_id in seen_effects:
                continue

            dominant_actor = link.actors[0] if link.actors else None
            if (
                dominant_actor
                and actor_counts[dominant_actor] >= _MAX_QUESTIONS_PER_ACTOR
            ):
                continue
            if (
                event_type_counts[link.cause_event_type]
                >= _MAX_QUESTIONS_PER_EVENT_TYPE
            ):
                continue

            question = self._build_counterfactual_question(
                link, all_artifact_ids, effect_cause_map
            )
            if question:
                questions.append(question)
                seen_effects.add(link.effect_event_id)
                if dominant_actor:
                    actor_counts[dominant_actor] += 1
                event_type_counts[link.cause_event_type] += 1

        return questions[: self.MAX_COUNTERFACTUAL]

    def _build_counterfactual_question(
        self,
        link: CausalLink,
        all_artifact_ids: List[str],
        effect_cause_map: Dict[str, List[CausalLink]],
    ) -> Optional[EvalQuestion]:
        question_text = self._prose_counterfactual(link)
        if not question_text:
            return None

        primary_actor = link.actors[0] if link.actors else None
        actor_role = None
        actor_visible = None
        actor_subsystems = None
        if primary_actor and self._policy:
            actor_role = self._policy.role_for_actor(primary_actor)
            if self._corpus:
                actor_visible = sorted(
                    self._policy.visible_artifacts(
                        actor_id=primary_actor,
                        all_artifact_ids=all_artifact_ids,
                        as_of=link.effect_timestamp,
                        corpus=self._corpus,
                    )
                )
            if actor_role:
                actor_subsystems = sorted(self._policy.subsystems_for_role(actor_role))

        # A link is a negative counterfactual (outcome_changed=False) when the
        # effect has multiple causes — removing one cause wouldn't stop the effect.
        is_multi_cause = len(effect_cause_map.get(link.effect_event_id, [])) >= 2

        ground_truth = {
            "outcome_changed": link.outcome_changed,
            "causal_mechanism": link.link_type,
            "mechanism_aliases": list(link.mechanism_aliases or []),
            "cause_event_id": link.cause_event_id,
            "cause_event_type": link.cause_event_type,
            "effect_event_id": link.effect_event_id,
            "effect_event_type": link.effect_event_type,
            "mechanism_direction": "cause_to_effect",
            "actors": list(link.actors or []),
            "evidence_artifacts": list(link.evidence_artifact_ids or []),
            "counterfactual_premise": link.counterfactual_premise,
            "counterfactual_outcome": link.counterfactual_outcome,
            "is_multi_cause_effect": is_multi_cause,
        }

        difficulty = (
            "hard"
            if len(link.actors) > 2
            else "medium"
            if link.outcome_changed
            else "easy"
        )

        qid = self._make_id(
            "C", link.link_type, link.cause_timestamp, link.effect_timestamp
        )
        return EvalQuestion(
            question_id=qid,
            question_type="COUNTERFACTUAL",
            question_text=question_text,
            difficulty=difficulty,
            ground_truth=ground_truth,
            actor=primary_actor,
            actor_role=actor_role,
            as_of_time=link.effect_timestamp,
            actor_visible_artifacts=actor_visible,
            actor_subsystem_access=actor_subsystems,
            causal_link=self._causal_link_contract(link),
            expected_answer_schema=ANSWER_SCHEMAS["COUNTERFACTUAL"],
        )

    def _prose_counterfactual(self, link: CausalLink) -> Optional[str]:
        cause_date = link.cause_timestamp[:10] if link.cause_timestamp else "that date"
        effect_date = (
            link.effect_timestamp[:10] if link.effect_timestamp else cause_date
        )

        external_actors = []
        internal_actors = []
        for actor in link.actors:
            role = self._policy.role_for_actor(actor) if self._policy else None
            if role == "external":
                external_actors.append(actor)
            else:
                internal_actors.append(actor)

        if external_actors and internal_actors:
            actors_str = (
                f"internal ({', '.join(internal_actors[:3])}) and "
                f"external ({', '.join(external_actors[:2])})"
            )
        elif link.actors:
            actors_str = ", ".join(link.actors[:3])
        else:
            actors_str = "the involved parties"

        ext_clause = ""
        if external_actors:
            ext_clause = f" (via {', '.join(external_actors[:2])})"

        domain_hint = (
            f"a {link.cause_event_type.replace('_', ' ')} and its downstream "
            f"effect ({link.effect_event_type.replace('_', ' ')})"
        )

        is_negative = not link.outcome_changed
        style_bank = (
            _NEGATIVE_COUNTERFACTUAL_STYLES if is_negative else _COUNTERFACTUAL_STYLES
        )
        style = random.choice(style_bank)

        deterministic = (
            f"Regarding the events between {cause_date} and {effect_date} "
            f"involving {actors_str}: "
            f"if {link.counterfactual_premise}{ext_clause}, "
            f"is it true that {link.counterfactual_outcome}?"
        )

        if not self._llm_fn:
            return deterministic

        cause_label = link.cause_event_type.replace("_", " ")
        effect_label = link.effect_event_type.replace("_", " ")

        if is_negative:
            context_note = (
                f"Context for phrasing only (do NOT restate): "
                f"there was a {cause_label} involving {actors_str} around {cause_date}, "
                f"and a {effect_label} occurred around {effect_date}. "
                f"But the {effect_label} had multiple contributing factors, "
                f"so removing just this one {cause_label} would not have prevented it. "
                f"Do NOT reveal this in the question — the question should ask whether "
                f"one event was truly dependent on the other or merely coincident."
            )
        else:
            context_note = (
                f"Context for phrasing only (do NOT restate): "
                f"there was a {cause_label} involving {actors_str} around {cause_date}, "
                f"and a {effect_label} occurred around {effect_date}. "
                f"Do NOT name or describe what the {effect_label} was — "
                f"the question should ask whether the {cause_label} was a necessary "
                f"precondition for what followed."
            )

        prompt = (
            f"{style}\n\n"
            f"{context_note}\n\n"
            f"Requirements:\n"
            f"- The question MUST end with a question mark.\n"
            f"- It MUST explicitly reference the date ({cause_date}) and at least one actor ({actors_str}).\n"
            f"- Do NOT name artifact IDs, ticket numbers, or document IDs.\n"
            f"- Do NOT use the causal mechanism label '{link.link_type}'.\n"
            f"- Do NOT reveal whether the answer is yes or no.\n"
            f"- Do NOT restate the context as a double-negative hypothetical.\n"
            f"- The question should sound like an investigator probing a dependency, "
            f"not someone who already knows what happened.\n"
            f"- Do NOT describe the {effect_label} in any specificity — "
            f"only refer to it indirectly (e.g., 'what happened next', 'the subsequent event').\n"
            f"- Use natural language. 15-100 words.\n"
            f"Output only the question text."
        )

        for attempt in range(3):
            retry_note = (
                " Previous attempt failed validation. Ensure the question: "
                "ends with '?', avoids artifact IDs, is 15-100 words, "
                "contains a counterfactual keyword (if/had/would/could/might), "
                "names specific actors instead of generic placeholders, "
                "and does not restate the premise and outcome as a double-negative."
                if attempt > 0
                else ""
            )
            try:
                result = self._llm_fn(prompt + retry_note).strip()
                if self._validate_prose(
                    result,
                    ground_truth_str=link.counterfactual_outcome,
                    question_type="COUNTERFACTUAL",
                    link=link,
                ):
                    return result
                else:
                    logger.debug(
                        f"[question_gen] Counterfactual prose validation failed "
                        f"(attempt {attempt + 1}): {result[:80]}"
                    )
            except Exception as exc:
                logger.warning(
                    f"[question_gen] Counterfactual prose generation error "
                    f"(attempt {attempt + 1}): {exc}"
                )

        return deterministic

    def _silence_questions(self) -> List[EvalQuestion]:
        """
        Builds SILENCE questions with stratified sampling across trigger types.

        Stratification prevents question clustering — e.g. all questions being
        "was a postmortem written after incident X/Y/Z?" when incidents are the
        most common trigger. One question is picked per (trigger_type, response_type)
        pair before filling the remainder at random.

        The pool is split evenly between exists=False (absence confirmed) and
        exists=True (response confirmed present).
        """
        target_false = self.MAX_SILENCE // 2
        target_true = self.MAX_SILENCE - target_false

        # --- Stratified sampling for false (absence) questions ---
        by_trigger: Dict[str, List[AbsenceRecord]] = defaultdict(list)
        for record in self._absences:
            by_trigger[record.trigger_event_type].append(record)

        n_triggers = len(by_trigger)
        per_trigger = max(1, target_false // max(n_triggers, 1))

        stratified: List[AbsenceRecord] = []
        for records in by_trigger.values():
            random.shuffle(records)
            by_response: Dict[str, List[AbsenceRecord]] = defaultdict(list)
            for r in records:
                by_response[r.expected_response_type].append(r)
            chosen: List[AbsenceRecord] = []
            response_groups = list(by_response.values())
            random.shuffle(response_groups)
            for group in response_groups:
                if len(chosen) >= per_trigger:
                    break
                chosen.append(random.choice(group))
            stratified.extend(chosen)

        selected_ids = {id(r) for r in stratified}
        remainder = [r for r in self._absences if id(r) not in selected_ids]
        if len(stratified) < target_false and remainder:
            extra = random.sample(
                remainder, min(target_false - len(stratified), len(remainder))
            )
            stratified.extend(extra)

        random.shuffle(stratified)
        false_pool = stratified[: target_false + 5]

        # --- Confirmed (true) questions ---
        confirmed_pool = list(self._confirmed)
        random.shuffle(confirmed_pool)
        confirmed_pool = confirmed_pool[: target_true + 5]

        false_questions: List[EvalQuestion] = []
        for record in false_pool:
            if len(false_questions) >= target_false:
                break
            q = self._build_silence_question(record, expected_exists=False)
            if q:
                false_questions.append(q)

        true_questions: List[EvalQuestion] = []
        for event in confirmed_pool:
            if len(true_questions) >= target_true:
                break
            q = self._build_silence_confirmed_question(event)
            if q:
                true_questions.append(q)

        questions = false_questions + true_questions
        random.shuffle(questions)
        return questions[: self.MAX_SILENCE]

    def _build_silence_question(
        self, record: AbsenceRecord, expected_exists: bool
    ) -> Optional[EvalQuestion]:
        """Build a SILENCE question for a verified absence."""
        # Skip questions where search space contains only generic subsystem names.
        # An agent that answers "no" without checking specific artifacts scores 0
        # on trajectory, so the question is uninformative if no real IDs exist.
        effective_space = [
            e
            for e in record.expected_search_space
            if e not in self._trivial_search_space
        ]
        if not effective_space:
            logger.debug(
                f"[question_gen] Skipping SILENCE question for "
                f"{record.trigger_event_id} — expected_search_space is empty "
                f"or contains only trivial entries"
            )
            return None

        question_text = self._prose_silence(record, expected_exists=expected_exists)
        if not question_text:
            return None

        ground_truth = {
            "exists": False,
            "absence_type": "event_log_confirmed",
            "reason": (
                f"No {record.expected_response_type.replace('_', ' ')} "
                f"followed the {record.trigger_event_type.replace('_', ' ')} "
                f"at {record.trigger_timestamp[:10]}"
            ),
            "trigger_event_id": record.trigger_event_id,
            "trigger_event_type": record.trigger_event_type,
            "expected_response_type": record.expected_response_type,
            "trigger_actors": record.trigger_actors,
            "expected_search_space": record.expected_search_space,
        }

        qid = self._make_id(
            "S0",
            record.trigger_event_type,
            record.trigger_timestamp,
            record.expected_response_type,
        )
        return EvalQuestion(
            question_id=qid,
            question_type="SILENCE",
            question_text=question_text,
            difficulty="hard",
            ground_truth=ground_truth,
            expected_search_space=record.expected_search_space,
            expected_answer_schema=ANSWER_SCHEMAS["SILENCE"],
        )

    def _build_silence_confirmed_question(
        self, event: LogEvent
    ) -> Optional[EvalQuestion]:
        """Build a SILENCE question for a confirmed response (exists=True)."""
        question_text = self._prose_silence_confirmed(event)
        if not question_text:
            return None

        artifacts = []
        for val in event.artifact_ids.values():
            if isinstance(val, list):
                artifacts.extend(str(v) for v in val if v)
            elif val:
                artifacts.append(str(val))

        ground_truth = {
            "exists": True,
            "absence_type": "event_log_present",
            "reason": f"Event {event.type} confirmed at {event.timestamp[:10]}",
            "artifact_ids": artifacts,
        }

        search_space = artifacts + (
            self._corpus.list_ids()[:10] if self._corpus else []
        )

        qid = self._make_id("S1", event.type, event.timestamp)
        return EvalQuestion(
            question_id=qid,
            question_type="SILENCE",
            question_text=question_text,
            difficulty="easy",
            ground_truth=ground_truth,
            expected_search_space=sorted(set(search_space)),
            expected_answer_schema=ANSWER_SCHEMAS["SILENCE"],
        )

    def _prose_silence(
        self, record: AbsenceRecord, expected_exists: bool
    ) -> Optional[str]:
        response_label = record.expected_response_type.replace("_", " ")
        trigger_label = record.trigger_event_type.replace("_", " ")

        external_actors = [
            a
            for a in record.trigger_actors
            if self._policy and self._policy.role_for_actor(a) == "external"
        ]
        if external_actors:
            trigger_party = f" from {', '.join(external_actors[:2])}"
        elif record.trigger_actors:
            trigger_party = f" involving {', '.join(record.trigger_actors[:2])}"
        else:
            trigger_party = ""

        template = (
            f"Was a {response_label} created following the "
            f"{trigger_label}{trigger_party} on {record.trigger_timestamp[:10]}?"
        )

        if not self._llm_fn:
            return template

        style = random.choice(_SILENCE_STYLES)
        actors_str = (
            ", ".join(record.trigger_actors[:3])
            if record.trigger_actors
            else "the involved parties"
        )

        prompt = (
            f"{style}\n\n"
            f"Ask whether a {response_label} was created following the "
            f"{trigger_label} on {record.trigger_timestamp[:10]} "
            f"involving {actors_str}.\n\n"
            f"Requirements:\n"
            f"- The question MUST end with a question mark.\n"
            f"- It MUST require investigation; do not imply the answer.\n"
            f"- Do NOT use normative phrases like 'as required by', 'should have been', "
            f"'was supposed to', 'procedure requires', 'protocol requires', 'per the', "
            f"'as mandated' — these leak the answer.\n"
            f"- Do NOT include artifact IDs or internal references.\n"
            f"- Use natural language. 10-60 words.\n"
            f"Output only the question text."
        )

        for attempt in range(3):
            retry_note = (
                " Previous attempt failed validation. Ensure the question ends with '?', "
                "names specific actors instead of generic placeholders, "
                "avoids answer-leaking phrases, and is 10-60 words."
                if attempt > 0
                else ""
            )
            try:
                result = self._llm_fn(prompt + retry_note).strip()
                if self._validate_prose(
                    result,
                    ground_truth_str=str(expected_exists),
                    question_type="SILENCE",
                ):
                    return result
                else:
                    logger.debug(
                        f"[question_gen] Silence prose validation failed "
                        f"(attempt {attempt + 1}): {result[:80]}"
                    )
            except Exception as exc:
                logger.warning(
                    f"[question_gen] Silence prose generation error "
                    f"(attempt {attempt + 1}): {exc}"
                )

        return template

    def _prose_silence_confirmed(self, event: LogEvent) -> Optional[str]:
        label = event.type.replace("_", " ")
        actors_str = (
            ", ".join(event.actors[:3]) if event.actors else "the involved parties"
        )

        template = f"Did a {label} occur on {event.timestamp[:10]}?"

        if not self._llm_fn:
            return template

        style = random.choice(_SILENCE_STYLES)

        prompt = (
            f"{style}\n\n"
            f"Ask whether a {label} occurred on {event.timestamp[:10]} "
            f"involving {actors_str}.\n\n"
            f"Requirements:\n"
            f"- The question MUST end with a question mark.\n"
            f"- It MUST require investigation; do not imply the answer.\n"
            f"- Do NOT use normative phrases like 'as required by', 'should have been', "
            f"'was supposed to', 'procedure requires', 'protocol requires', 'per the', "
            f"'as mandated' -- these leak the answer.\n"
            f"- Do NOT include artifact IDs or internal references.\n"
            f"- Use natural language. 10-60 words.\n"
            f"Output only the question text."
        )

        for attempt in range(3):
            retry_note = (
                " Previous attempt failed validation. Ensure the question ends with '?', "
                "names specific actors instead of generic placeholders, "
                "avoids answer-leaking phrases, and is 10-60 words."
                if attempt > 0
                else ""
            )
            try:
                result = self._llm_fn(prompt + retry_note).strip()
                if self._validate_prose(
                    result,
                    ground_truth_str="True",
                    question_type="SILENCE",
                ):
                    return result
                else:
                    logger.debug(
                        f"[question_gen] Silence confirmed prose validation failed "
                        f"(attempt {attempt + 1}): {result[:80]}"
                    )
            except Exception as exc:
                logger.warning(
                    f"[question_gen] Silence confirmed prose generation error "
                    f"(attempt {attempt + 1}): {exc}"
                )

        return template

    def _validate_prose(
        self,
        text: str,
        ground_truth_str: str,
        question_type: str,
        link: Optional[CausalLink] = None,
    ) -> bool:
        """
        Validate LLM-generated question prose against a structured rubric.

        Rules applied to all question types:
          1. Must end with '?'
          2. Must be 10-150 words
          3. Must not contain raw artifact IDs (XX-\\d+ or [a-f0-9]{8,})
          4. Must not leak ground truth verbatim (if ground_truth_str is non-trivial)

        Additional rules per type:

        PERSPECTIVE
          5. Must contain a temporal anchor (date, "as of", or named actor reference)

        COUNTERFACTUAL
          5. Must contain a counterfactual keyword (if/had/would/could/might/hypothetically)
          6. Must not be a circular double-negative (premise == outcome rephrased)

        SILENCE
          5. Must contain an existence verb (was/were/did/has/have/is/are)
          6. Must not contain answer-leaking normative phrases
        """
        normalised = (
            text
            .replace("\u2010", "-")
            .replace("\u2011", "-")
            .replace("\u2012", "-")
            .replace("\u2013", "-")
        )

        # Rule 1: ends with '?'
        if not normalised.endswith("?"):
            return False

        # Rule 2: word count
        words = normalised.split()
        if len(words) < 10 or len(words) > 150:
            return False

        # Rule 3: no raw artifact IDs
        if re.search(r"\b[A-Z]{1,4}-\d{2,6}\b", normalised):
            return False
        if re.search(r"\b[a-f0-9]{8,}\b", normalised):
            return False

        # Rule 4: no verbatim ground truth leak (skip very short or empty values)
        if ground_truth_str and len(ground_truth_str) > 4:
            if ground_truth_str.lower() in normalised.lower():
                return False

        # Type-specific rules
        if question_type == "PERSPECTIVE":
            if not re.search(
                r"\d{4}-\d{2}-\d{2}|as of|by\s+[A-Z][a-z]+", normalised, re.IGNORECASE
            ):
                return False

        elif question_type == "COUNTERFACTUAL":
            # Must contain a counterfactual keyword.
            if not re.search(
                r"\b(if|had|would|could|might|hypothetically)\b",
                normalised,
                re.IGNORECASE,
            ):
                return False

            # Must not be a circular double-negative where premise and outcome
            # are effectively the same clause (the original bug: "if the incident
            # had not occurred, would the postmortem not have been written and
            # the root cause analysis would not exist?").
            if link is not None:
                premise_words = set(link.counterfactual_premise.lower().split())
                outcome_words = set(link.counterfactual_outcome.lower().split())
                # High overlap between premise and outcome word sets indicates
                # the LLM just repeated both halves of the link verbatim.
                if len(premise_words) > 3 and len(outcome_words) > 3:
                    overlap = premise_words & outcome_words
                    stopwords = {
                        "the",
                        "a",
                        "an",
                        "and",
                        "or",
                        "not",
                        "have",
                        "has",
                        "been",
                        "would",
                        "had",
                        "if",
                        "is",
                        "was",
                        "were",
                        "will",
                        "that",
                        "this",
                        "it",
                        "in",
                        "of",
                        "to",
                        "for",
                    }
                    meaningful_overlap = overlap - stopwords
                    overlap_ratio = len(meaningful_overlap) / max(
                        len(premise_words - stopwords), 1
                    )
                    if overlap_ratio > 0.7:
                        logger.debug(
                            f"[question_gen] Rejecting circular counterfactual "
                            f"(premise/outcome overlap ratio {overlap_ratio:.2f}): "
                            f"{text[:80]}"
                        )
                        return False

        elif question_type == "SILENCE":
            # Must contain an existence verb.
            if not re.search(
                r"\b(was|were|did|has|have|is|are)\b", normalised, re.IGNORECASE
            ):
                return False

            _SILENCE_LEAK_PHRASES = (
                "as required by",
                "as expected by",
                "should have been",
                "was supposed to",
                "procedure requires",
                "protocol requires",
                "per the",
                "as mandated",
            )
            if any(phrase in normalised.lower() for phrase in _SILENCE_LEAK_PHRASES):
                return False

        return True

    def _infer_actors(self) -> List[str]:
        seen: Dict[str, int] = {}
        for event in self._events:
            for actor in event.actors:
                seen[actor] = seen.get(actor, 0) + 1
        return [
            a
            for a, _ in sorted(seen.items(), key=lambda x: -x[1])
            if not a.startswith("system") and a
        ][:10]

    def _artifact_subsystem(self, artifact_id: str) -> Optional[str]:
        if self._corpus:
            return self._corpus.subsystem_of(artifact_id)
        return None

    def _make_id(self, prefix: str, *parts: str) -> str:
        raw = prefix + "_" + "_".join(str(p)[:20] for p in parts)
        digest = hashlib.md5(raw.encode()).hexdigest()[:8]
        return f"{prefix}_{digest}"

    def _causal_link_contract(self, link: CausalLink) -> Dict[str, Any]:
        return {
            "link_type": link.link_type,
            "cause_event_id": link.cause_event_id,
            "cause_event_type": link.cause_event_type,
            "cause_timestamp": link.cause_timestamp,
            "effect_event_id": link.effect_event_id,
            "effect_event_type": link.effect_event_type,
            "effect_timestamp": link.effect_timestamp,
            "actors": list(link.actors or []),
            "counterfactual_premise": link.counterfactual_premise,
            "counterfactual_outcome": link.counterfactual_outcome,
            "outcome_changed": link.outcome_changed,
            "cause_artifact_ids": list(link.cause_artifact_ids or []),
            "effect_artifact_ids": list(link.effect_artifact_ids or []),
            "evidence_artifact_ids": list(link.evidence_artifact_ids or []),
            "mechanism_aliases": list(link.mechanism_aliases or []),
        }
