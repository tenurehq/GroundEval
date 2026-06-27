"""
groundeval/core.py
=======================
Protocol definitions and shared data models.

The framework separates three concerns:
  1. CorpusAdapter  — how to fetch artifacts (MongoDB, files, or nothing)
  2. AccessPolicy   — who can see what subsystem at what time
  3. TaskContract   — what the agent must verify before acting

Single command: groundeval task --config config.yaml
One config format: task_contracts with actors and roles.
One question type: task contract.
Three scoring tracks applied to every run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """A single tool call recorded by the GatedRuntime."""

    tool_name: str
    arguments: dict[str, Any]
    result_ids: list[str]
    timestamp_applied: str | None
    horizon_violation: bool
    actor_gate_violation: bool
    subsystem_violation: bool
    returned_empty: bool
    latency_ms: float


@dataclass
class AgentTrajectory:
    """Complete trace of one agent run against a task."""

    task_id: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    cited_artifacts: list[str] = field(default_factory=list)
    final_answer: dict[str, Any] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    horizon_violations: int = 0
    actor_gate_violations: int = 0
    subsystem_violations: int = 0
    dead_ends_hit: int = 0
    dead_ends_recovered: int = 0
    budget_exceeded: bool = False


@dataclass
class TaskPrecondition:
    """
    A single precondition the agent must verify before the task action is valid.

    check: short name for this precondition (e.g. "recipient_is_active_customer")
    description: human-readable description of what must be verified
    required_facts: list of fact keys that must be present in the agent's answer
    ground_truth_field: dotted path into the seeded artifacts for the correct value
                        (e.g. "crm_account.account_status")
    """

    check: str
    description: str
    required_facts: list[str] = field(default_factory=list)
    ground_truth_field: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "TaskPrecondition":
        return cls(
            check=d["check"],
            description=d.get("description", d["check"]),
            required_facts=list(d.get("required_facts", [])),
            ground_truth_field=d.get("ground_truth_field", ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AllowedTool:
    """
    Declaration of an allowed tool in minimal-fixture mode.

    tool_name: name of the CrewAI tool (e.g. "fetch_customer")
    entity_arg: the argument that carries the entity ID
    returns: dict of field -> value to populate in the tool's return
    action: bool, True if calling this tool counts as "acting"
    """

    tool_name: str
    entity_arg: str = ""
    returns: dict[str, Any] = field(default_factory=dict)
    action: bool = False
    artifact_id: str = ""
    subsystem: str = ""
    timestamp: str = ""

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "AllowedTool":
        return cls(
            tool_name=name,
            entity_arg=d.get("entity_arg", ""),
            returns=dict(d.get("returns", {})),
            action=d.get("action", False),
            artifact_id=d.get("artifact_id", ""),
            subsystem=d.get("subsystem", ""),
            timestamp=d.get("timestamp", ""),
        )

    def to_dict(self) -> dict:
        return {
            "entity_arg": self.entity_arg,
            "returns": self.returns,
            "action": self.action,
            "artifact_id": self.artifact_id,
            "subsystem": self.subsystem,
            "timestamp": self.timestamp,
        }


@dataclass
class TaskContract:
    name: str
    task_description: str
    preconditions: list[TaskPrecondition]
    valid_action: str = "all_preconditions_pass"
    decision_field: str = "should_act"
    artifacts_dir: str = "./data"
    actor: str | None = None
    role: str | None = None
    actors: dict[str, str] = field(default_factory=dict)
    roles: dict[str, dict] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[AllowedTool] = field(default_factory=list)
    expected_action: bool | None = None
    action_tool: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "TaskContract":
        preconditions = [
            TaskPrecondition.from_dict(pc) for pc in d.get("preconditions", [])
        ]
        allowed_tools_raw = d.get("allowed_tools", {})
        allowed_tools = [
            AllowedTool.from_dict(name, tool_cfg)
            for name, tool_cfg in allowed_tools_raw.items()
        ]
        return cls(
            name=d["name"],
            task_description=d.get("task_description", d["name"]),
            preconditions=preconditions,
            valid_action=d.get("valid_action", "all_preconditions_pass"),
            decision_field=d.get("decision_field", "should_act"),
            artifacts_dir=d.get("artifacts_dir", "./data"),
            actor=d.get("actor"),
            role=d.get("role"),
            actors=dict(d.get("actors", {})),
            roles=dict(d.get("roles", {})),
            inputs=dict(d.get("inputs", {})),
            allowed_tools=allowed_tools,
            expected_action=d.get("expected_action"),
            action_tool=d.get("action_tool", ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_fixture_mode(self) -> bool:
        return bool(self.allowed_tools)


# ──────────────────────────────────────────────────────────────
# Evaluation Result
# ──────────────────────────────────────────────────────────────


@dataclass
class TaskEvalResult:
    """
    The complete result of scoring one task run through all three tracks.

    Per-track scores (counterfactual_score, silence_score, perspective_score)
    plus an overall that is the mean of the three.
    """

    task_name: str
    # Per-track scores
    counterfactual_score: float
    silence_score: float
    perspective_score: float
    # Overall (mean of the three track scores)
    overall_score: float
    # Answer details
    answer_correct: bool
    precondition_results: list[dict[str, Any]] = field(default_factory=list)
    # Violations
    horizon_violations: int = 0
    actor_gate_violations: int = 0
    subsystem_violations: int = 0
    # Trajectory details
    dead_ends_hit: int = 0
    dead_ends_recovered: int = 0
    tool_call_count: int = 0
    # Tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Budget
    budget_exceeded: bool = False
    # Extra
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_violations"] = (
            self.horizon_violations
            + self.actor_gate_violations
            + self.subsystem_violations
        )
        return d


# ──────────────────────────────────────────────────────────────
# Protocols
# ──────────────────────────────────────────────────────────────


@runtime_checkable
class CorpusAdapter(Protocol):
    """
    Provides artifact retrieval for the gated tool layer.

    Two built-in implementations are provided:
      - FileCorpusAdapter   — artifacts are files in a directory
      - NullCorpusAdapter   — context-injection mode, no retrieval needed

    Implement this protocol to plug in MongoDB, Elasticsearch, etc.
    """

    def fetch(self, artifact_id: str, as_of: str | None = None) -> dict | None:
        """Return artifact dict or None if not found / gated by timestamp."""
        ...

    def search(
        self,
        query: str,
        artifact_type: str | None = None,
        as_of: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Full-text or keyword search. Return list of artifact dicts."""
        ...

    def timestamp_of(self, artifact_id: str) -> str | None:
        """Return ISO timestamp of artifact creation, or None."""
        ...

    def subsystem_of(self, artifact_id: str) -> str | None:
        """Return subsystem name for this artifact (e.g. 'email', 'jira')."""
        ...

    def list_ids(self, subsystem: str | None = None) -> list[str]:
        """List all artifact IDs, optionally filtered by subsystem."""
        ...


@runtime_checkable
class AccessPolicy(Protocol):
    """
    Declares who can see what.

    The simplest implementation reads from config.yaml:

        roles:
          engineer:
            subsystems: [jira, git, slack, confluence]
          sales:
            subsystems: [salesforce, email, slack]

    Implement this protocol for more dynamic rules (e.g. per-project access).
    """

    def subsystems_for_role(self, role: str) -> set[str]:
        """Return set of subsystem names accessible to this role."""
        ...

    def role_for_actor(self, actor_id: str) -> str | None:
        """Return the role name for this actor, or None if unknown."""
        ...

    def visible_artifacts(
        self,
        actor_id: str,
        all_artifact_ids: list[str],
        as_of: str | None = None,
        corpus: CorpusAdapter | None = None,
    ) -> set[str]:
        """
        Return the subset of artifact IDs visible to this actor.

        Default implementation: all artifacts whose subsystem is in the
        actor's role subsystem set. Override for finer-grained control
        (e.g. direct-involvement tracking from an event log).
        """
        ...


class GatedRuntime:
    """
    First-class trace-capture runtime.

    Wraps a CorpusAdapter + AccessPolicy and enforces gates:
      - Temporal gate (as_of_time): artifacts created after this time are invisible
      - Actor gate (visibility cone): only artifacts the actor can see
      - Subsystem gate: artifacts outside the actor's role subsystems are blocked

    Exposed methods mirror ``CorpusAdapter`` but emit gated results, record
    every call in a ``ToolCall`` log, and build a live ``AgentTrajectory``.

    Usage::

        runtime = GatedRuntime(
            corpus=corpus,
            policy=policy,
            task_id="verify_recipient",
            actor="alice",
            as_of="2026-01-15T09:00:00",
            actor_visible_artifacts={"email-001", "crm-002"},
            actor_subsystem_access={"email", "crm"},
        )
        doc = runtime.fetch("email-001")          # returns None if gated out
        results = runtime.search("foo", limit=5)  # gated search
        traj = runtime.trajectory()               # current trace so far

    All violations are recorded internally; the scorer later consumes them.
    """

    def __init__(
        self,
        corpus: CorpusAdapter,
        policy: AccessPolicy,
        task_id: str,
        actor: str | None = None,
        as_of: str | None = None,
        actor_visible_artifacts: set[str] | None = None,
        actor_subsystem_access: set[str] | None = None,
    ):
        self._corpus = corpus
        self._policy = policy
        self._task_id = task_id
        self._actor = actor
        self._as_of = as_of
        self._actor_visible = actor_visible_artifacts or set()
        self._actor_subsystems = actor_subsystem_access or set()
        self._call_log: list[ToolCall] = []
        self._trajectory = AgentTrajectory(task_id=task_id)
        self._all_subsystems: list[str] = []

    @property
    def call_log(self) -> list[ToolCall]:
        return list(self._call_log)

    @property
    def all_subsystems(self) -> list[str]:
        return list(self._all_subsystems)

    def trajectory(self) -> AgentTrajectory:
        """Return the live trajectory populated so far."""
        t = self._trajectory
        t.tool_calls = list(self._call_log)
        t.horizon_violations = sum(1 for c in self._call_log if c.horizon_violation)
        t.actor_gate_violations = sum(
            1 for c in self._call_log if c.actor_gate_violation
        )
        t.subsystem_violations = sum(1 for c in self._call_log if c.subsystem_violation)
        t.dead_ends_hit = sum(1 for c in self._call_log if c.returned_empty)
        recovered = 0
        for i in range(len(self._call_log) - 1):
            if (
                self._call_log[i].returned_empty
                and not self._call_log[i + 1].returned_empty
            ):
                recovered += 1
        t.dead_ends_recovered = recovered
        return t

    def fetch(self, artifact_id: str) -> dict | None:
        """Return artifact dict or None if gated."""
        import time

        t0 = time.time()

        doc = self._corpus.fetch(artifact_id)
        horizon = False
        if self._as_of and doc:
            ts = self._corpus.timestamp_of(artifact_id)
            if ts and ts > self._as_of:
                doc = None
                horizon = True

        actor_v, sub_v = False, False
        if doc:
            sub = self._corpus.subsystem_of(artifact_id)
            if self._actor_subsystems and sub and sub not in self._actor_subsystems:
                sub_v = True
                doc = None
            if self._actor_visible and artifact_id not in self._actor_visible:
                actor_v = True
                doc = None

        filtered = self._record(
            tool_name="fetch_artifact",
            arguments={"artifact_id": artifact_id},
            results=[doc] if doc else [],
            t0=t0,
            horizon_violation=horizon,
            actor_gate_violation=actor_v,
            subsystem_violation=sub_v,
            timestamp_applied=self._as_of,
        )
        return filtered[0] if filtered else None

    _SEARCH_INCLUDE_FIELDS = frozenset({
        "id",
        "_id",
        "type",
        "title",
        "timestamp",
        "date",
        "created_at",
        "actors",
        "participants",
        "subsystem",
        "summary",
        "description",
    })

    def search(
        self,
        query: str,
        artifact_type: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Gated search — results are post-filtered by as_of and actor cone.

        Search results are stripped to metadata-only. When the agent wants
        full content, it must call fetch_artifact().
        """
        import time, copy

        t0 = time.time()
        raw = self._corpus.search(
            query, artifact_type=artifact_type, as_of=self._as_of, limit=limit
        )
        stripped = []

        for doc in raw:
            doc_copy = {
                k: v for k, v in doc.items() if k in self._SEARCH_INCLUDE_FIELDS
            }
            stripped.append(doc_copy)

        sub_v = False
        if (
            artifact_type
            and self._actor_subsystems
            and artifact_type not in self._actor_subsystems
        ):
            sub_v = True
            return self._record(
                tool_name="search_artifacts",
                arguments={
                    "query": query,
                    "artifact_type": artifact_type,
                    "limit": limit,
                },
                results=[],
                t0=t0,
                horizon_violation=False,
                actor_gate_violation=False,
                subsystem_violation=sub_v,
                timestamp_applied=self._as_of,
            )

        return self._record(
            tool_name="search_artifacts",
            arguments={"query": query, "artifact_type": artifact_type, "limit": limit},
            results=stripped,
            t0=t0,
            horizon_violation=False,
            actor_gate_violation=False,
            subsystem_violation=sub_v,
            timestamp_applied=self._as_of,
        )

    def timestamp_of(self, artifact_id: str) -> str | None:
        return self._corpus.timestamp_of(artifact_id)

    def subsystem_of(self, artifact_id: str) -> str | None:
        return self._corpus.subsystem_of(artifact_id)

    def list_ids(self, subsystem: str | None = None) -> list[str]:
        if self._actor_visible:
            all_ids = self._corpus.list_ids(subsystem=subsystem)
            return [aid for aid in all_ids if aid in self._actor_visible]
        if (
            subsystem
            and self._actor_subsystems
            and subsystem not in self._actor_subsystems
        ):
            return []
        return self._corpus.list_ids(subsystem=subsystem)

    def _record(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        results: list[dict],
        t0: float,
        horizon_violation: bool = False,
        actor_gate_violation: bool = False,
        subsystem_violation: bool = False,
        timestamp_applied: str | None = None,
    ) -> list[dict]:
        import time

        latency = (time.time() - t0) * 1000

        filtered: list[dict] = []
        for r in results:
            ts = r.get("timestamp") or r.get("created_at") or r.get("date", "")
            if self._as_of and ts and ts > self._as_of:
                horizon_violation = True
                continue
            filtered.append(r)

        for r in filtered:
            doc_id = str(r.get("id", r.get("_id", "")))
            subsystem = r.get("subsystem", "")
            if self._actor_visible and doc_id and doc_id not in self._actor_visible:
                actor_gate_violation = True
            if (
                self._actor_subsystems
                and subsystem
                and subsystem not in self._actor_subsystems
            ):
                subsystem_violation = True

        tool_subsystem = arguments.get("artifact_type")
        if (
            tool_subsystem
            and self._actor_subsystems
            and tool_subsystem not in self._actor_subsystems
        ):
            subsystem_violation = True

        result_ids = [str(r.get("id", r.get("_id", ""))) for r in filtered]
        self._call_log.append(
            ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result_ids=result_ids,
                timestamp_applied=timestamp_applied,
                horizon_violation=horizon_violation,
                actor_gate_violation=actor_gate_violation,
                subsystem_violation=subsystem_violation,
                returned_empty=len(filtered) == 0,
                latency_ms=latency,
            )
        )
        return filtered


def _get_nested(d: dict, dotted_path: str) -> Any:
    """Retrieve a value from a nested dict or list using a dotted path string.
    Supports integer list indices, e.g. 'actors.0'."""
    parts = dotted_path.split(".")
    cur = d
    for p in parts:
        if isinstance(cur, dict):
            cur = cur.get(p)
        elif isinstance(cur, list):
            try:
                idx = int(p)
                cur = cur[idx] if 0 <= idx < len(cur) else None
            except ValueError:
                return None
        else:
            return None
    return cur


class FixtureBackend:
    """
    CorpusAdapter-compatible backend that synthesizes data from task contract
    declarations instead of loading from a file corpus.

    Each AllowedTool in the contract becomes a virtual artifact. The artifact
    ID is taken from AllowedTool.artifact_id (falls back to tool_name).
    fetch() returns the declared returns dict with id, subsystem, and timestamp
    set from the AllowedTool declaration when available.
    search() does substring matching across declared artifact metadata.

    Implements the same surface as CorpusAdapter so it slots into GatedRuntime
    without any runtime changes.
    """

    def __init__(self, allowed_tools: list[AllowedTool]):
        self._by_id: dict[str, AllowedTool] = {}
        for t in allowed_tools:
            aid = t.artifact_id or t.tool_name
            if aid in self._by_id:
                import warnings

                warnings.warn(
                    f"FixtureBackend: artifact_id '{aid}' declared by "
                    f"tool '{t.tool_name}' overwrites an existing entry "
                    f"from tool '{self._by_id[aid].tool_name}'. "
                    f"Check for duplicate artifact_id values across allowed_tools.",
                    stacklevel=2,
                )
            self._by_id[aid] = t

    def fetch(self, artifact_id: str, as_of: str | None = None) -> dict | None:
        tool = self._by_id.get(artifact_id)
        if tool is None:
            return None
        if as_of and tool.timestamp and tool.timestamp > as_of:
            return None
        doc = dict(tool.returns)
        if tool.subsystem:
            doc.setdefault("subsystem", tool.subsystem)
        if tool.timestamp:
            doc.setdefault("timestamp", tool.timestamp)
        doc.setdefault("id", artifact_id)
        return doc

    def search(
        self,
        query: str,
        artifact_type: str | None = None,
        as_of: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        import re

        if limit <= 0:
            return []

        pattern = re.compile(re.escape(query), re.IGNORECASE)
        results: list[dict] = []
        for aid, tool in self._by_id.items():
            if artifact_type and tool.subsystem != artifact_type:
                continue
            if as_of and tool.timestamp and tool.timestamp > as_of:
                continue
            doc = dict(tool.returns)
            doc.setdefault("id", aid)
            if tool.subsystem:
                doc.setdefault("subsystem", tool.subsystem)
            if tool.timestamp:
                doc.setdefault("timestamp", tool.timestamp)
            if pattern.search(json.dumps(doc)):
                results.append(doc)
                if len(results) >= limit:
                    break
        return results

    def timestamp_of(self, artifact_id: str) -> str | None:
        tool = self._by_id.get(artifact_id)
        if tool is None:
            return None
        return tool.timestamp or None

    def subsystem_of(self, artifact_id: str) -> str | None:
        tool = self._by_id.get(artifact_id)
        if tool is None:
            return None
        return tool.subsystem or None

    def list_ids(self, subsystem: str | None = None) -> list[str]:
        if not subsystem:
            return list(self._by_id.keys())
        return [aid for aid, t in self._by_id.items() if t.subsystem == subsystem]


ANSWER_SCHEMA_TASK: dict[str, Any] = {
    "type": "object",
    "required": ["preconditions_verified", "all_preconditions_pass", "reasoning"],
    "properties": {
        "preconditions_verified": {
            "type": "array",
            "description": "List of precondition check results with facts found",
            "items": {
                "type": "object",
                "required": ["check", "passed", "facts_found"],
                "properties": {
                    "check": {"type": "string"},
                    "passed": {"type": "boolean"},
                    "facts_found": {"type": "object"},
                    "evidence_artifacts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "all_preconditions_pass": {"type": "boolean"},
        "should_act": {"type": "boolean"},
        "reasoning": {"type": "string"},
        "evidence_artifacts": {"type": "array", "items": {"type": "string"}},
    },
}
