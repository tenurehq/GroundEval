from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    result_ids: list[str]
    timestamp_applied: str | None
    horizon_violation: bool
    actor_gate_violation: bool
    subsystem_violation: bool
    returned_empty: bool
    latency_ms: float
    agent_name: str | None = None
    node_name: str | None = None
    workflow_run_id: str | None = None
    branch_id: str | None = None
    call_id: str | None = None
    parent_event_id: str | None = None


@dataclass
class AgentTrajectory:
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
    events: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolExpectation:
    tool: str
    match_args: dict[str, Any] = field(default_factory=dict)
    expected_return: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'ToolExpectation':
        return cls(
            tool=str(d.get('tool', '')),
            match_args=dict(d.get('match_args', {})),
            expected_return=dict(d.get('expected_return', {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskPrecondition:
    check: str
    description: str
    required_facts: list[str] = field(default_factory=list)
    ground_truth_field: str = ''
    required_tool: str = ''
    expected_field: str = ''

    @classmethod
    def from_dict(cls, d: dict) -> 'TaskPrecondition':
        return cls(
            check=d['check'],
            description=d.get('description', d['check']),
            required_facts=list(d.get('required_facts', [])),
            ground_truth_field=d.get('ground_truth_field', ''),
            required_tool=d.get('required_tool', ''),
            expected_field=d.get('expected_field', ''),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AllowedTool:
    tool_name: str
    entity_arg: str = ''
    returns: dict[str, Any] = field(default_factory=dict)
    action: bool = False
    artifact_id: str = ''
    subsystem: str = ''
    timestamp: str = ''

    @classmethod
    def from_dict(cls, name: str, d: dict) -> 'AllowedTool':
        return cls(
            tool_name=name,
            entity_arg=d.get('entity_arg', ''),
            returns=dict(d.get('returns', {})),
            action=d.get('action', False),
            artifact_id=d.get('artifact_id', ''),
            subsystem=d.get('subsystem', ''),
            timestamp=d.get('timestamp', ''),
        )

    def to_dict(self) -> dict:
        return {
            'entity_arg': self.entity_arg,
            'returns': self.returns,
            'action': self.action,
            'artifact_id': self.artifact_id,
            'subsystem': self.subsystem,
            'timestamp': self.timestamp,
        }


@dataclass
class TaskContract:
    name: str
    task_description: str
    preconditions: list[TaskPrecondition]
    valid_action: str = 'all_preconditions_pass'
    decision_field: str = 'should_act'
    artifacts_dir: str = './data'
    actor: str | None = None
    role: str | None = None
    actors: dict[str, str] = field(default_factory=dict)
    roles: dict[str, dict] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[AllowedTool] = field(default_factory=list)
    tool_expectations: list[ToolExpectation] = field(default_factory=list)
    expected_action: bool | None = None
    action_tool: str = ''
    required_agents: list[dict[str, Any]] = field(default_factory=list)
    required_handoffs: list[dict[str, Any]] = field(default_factory=list)
    required_agent_tool_expectations: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> 'TaskContract':
        preconditions = [
            TaskPrecondition.from_dict(pc) for pc in d.get('preconditions', [])
        ]
        allowed_tools_raw = d.get('allowed_tools', {})
        allowed_tools = [
            AllowedTool.from_dict(name, tool_cfg)
            for name, tool_cfg in allowed_tools_raw.items()
        ]
        tool_expectations = [
            ToolExpectation.from_dict(exp) for exp in d.get('tool_expectations', [])
        ]
        return cls(
            name=d['name'],
            task_description=d.get('task_description', d['name']),
            preconditions=preconditions,
            valid_action=d.get('valid_action', 'all_preconditions_pass'),
            decision_field=d.get('decision_field', 'should_act'),
            artifacts_dir=d.get('artifacts_dir', './data'),
            actor=d.get('actor'),
            role=d.get('role'),
            actors=dict(d.get('actors', {})),
            roles=dict(d.get('roles', {})),
            inputs=dict(d.get('inputs', {})),
            allowed_tools=allowed_tools,
            tool_expectations=tool_expectations,
            expected_action=d.get('expected_action'),
            action_tool=d.get('action_tool', ''),
            required_agents=list(d.get('required_agents', [])),
            required_handoffs=list(d.get('required_handoffs', [])),
            required_agent_tool_expectations=list(
                d.get('required_agent_tool_expectations', [])
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_fixture_mode(self) -> bool:
        return bool(self.allowed_tools)

    @property
    def is_framework_contract(self) -> bool:
        return bool(self.tool_expectations) or any(
            pc.required_tool or pc.expected_field for pc in self.preconditions
        )


@dataclass
class TaskEvalResult:
    task_name: str
    counterfactual_score: float
    silence_score: float
    perspective_score: float
    overall_score: float
    answer_correct: bool
    precondition_results: list[dict[str, Any]] = field(default_factory=list)
    horizon_violations: int = 0
    actor_gate_violations: int = 0
    subsystem_violations: int = 0
    dead_ends_hit: int = 0
    dead_ends_recovered: int = 0
    tool_call_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    budget_exceeded: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['total_violations'] = (
            self.horizon_violations
            + self.actor_gate_violations
            + self.subsystem_violations
        )
        return d


@runtime_checkable
class CorpusAdapter(Protocol):
    def fetch(self, artifact_id: str, as_of: str | None = None) -> dict | None: ...
    def search(self, query: str, artifact_type: str | None = None, as_of: str | None = None, limit: int = 10) -> list[dict]: ...
    def timestamp_of(self, artifact_id: str) -> str | None: ...
    def subsystem_of(self, artifact_id: str) -> str | None: ...
    def list_ids(self, subsystem: str | None = None) -> list[str]: ...


@runtime_checkable
class AccessPolicy(Protocol):
    def subsystems_for_role(self, role: str) -> set[str]: ...
    def role_for_actor(self, actor_id: str) -> str | None: ...
    def visible_artifacts(self, actor_id: str, all_artifact_ids: list[str], as_of: str | None = None, corpus: CorpusAdapter | None = None) -> set[str]: ...


class GatedRuntime:
    def __init__(self, corpus: CorpusAdapter, policy: AccessPolicy, task_id: str, actor: str | None = None, as_of: str | None = None, actor_visible_artifacts: set[str] | None = None, actor_subsystem_access: set[str] | None = None):
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
        t = self._trajectory
        t.tool_calls = list(self._call_log)
        t.horizon_violations = sum(1 for c in self._call_log if c.horizon_violation)
        t.actor_gate_violations = sum(1 for c in self._call_log if c.actor_gate_violation)
        t.subsystem_violations = sum(1 for c in self._call_log if c.subsystem_violation)
        t.dead_ends_hit = sum(1 for c in self._call_log if c.returned_empty)
        recovered = 0
        for i in range(len(self._call_log) - 1):
            if self._call_log[i].returned_empty and not self._call_log[i + 1].returned_empty:
                recovered += 1
        t.dead_ends_recovered = recovered
        return t

    def fetch(self, artifact_id: str) -> dict | None:
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
            tool_name='fetch_artifact',
            arguments={'artifact_id': artifact_id},
            results=[doc] if doc else [],
            t0=t0,
            horizon_violation=horizon,
            actor_gate_violation=actor_v,
            subsystem_violation=sub_v,
            timestamp_applied=self._as_of,
        )
        return filtered[0] if filtered else None

    _SEARCH_INCLUDE_FIELDS = frozenset({'id', '_id', 'type', 'title', 'timestamp', 'date', 'created_at', 'actors', 'participants', 'subsystem', 'summary', 'description'})

    def search(self, query: str, artifact_type: str | None = None, limit: int = 10) -> list[dict]:
        import time
        t0 = time.time()
        raw = self._corpus.search(query, artifact_type=artifact_type, as_of=self._as_of, limit=limit)
        stripped = []
        for doc in raw:
            doc_copy = {k: v for k, v in doc.items() if k in self._SEARCH_INCLUDE_FIELDS}
            stripped.append(doc_copy)
        sub_v = False
        if artifact_type and self._actor_subsystems and artifact_type not in self._actor_subsystems:
            sub_v = True
            return self._record(
                tool_name='search_artifacts',
                arguments={'query': query, 'artifact_type': artifact_type, 'limit': limit},
                results=[],
                t0=t0,
                horizon_violation=False,
                actor_gate_violation=False,
                subsystem_violation=sub_v,
                timestamp_applied=self._as_of,
            )
        return self._record(
            tool_name='search_artifacts',
            arguments={'query': query, 'artifact_type': artifact_type, 'limit': limit},
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
        if subsystem and self._actor_subsystems and subsystem not in self._actor_subsystems:
            return []
        return self._corpus.list_ids(subsystem=subsystem)

    def _record(self, tool_name: str, arguments: dict[str, Any], results: list[dict], t0: float, horizon_violation: bool = False, actor_gate_violation: bool = False, subsystem_violation: bool = False, timestamp_applied: str | None = None) -> list[dict]:
        import time
        latency = (time.time() - t0) * 1000
        filtered: list[dict] = []
        for r in results:
            ts = r.get('timestamp') or r.get('created_at') or r.get('date', '')
            if self._as_of and ts and ts > self._as_of:
                horizon_violation = True
                continue
            filtered.append(r)
        for r in filtered:
            doc_id = str(r.get('id', r.get('_id', '')))
            subsystem = r.get('subsystem', '')
            if self._actor_visible and doc_id and doc_id not in self._actor_visible:
                actor_gate_violation = True
            if self._actor_subsystems and subsystem and subsystem not in self._actor_subsystems:
                subsystem_violation = True
        tool_subsystem = arguments.get('artifact_type')
        if tool_subsystem and self._actor_subsystems and tool_subsystem not in self._actor_subsystems:
            subsystem_violation = True
        result_ids = [str(r.get('id', r.get('_id', ''))) for r in filtered]
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


class FixtureBackend:
    def __init__(self, allowed_tools: list[AllowedTool]):
        self._by_id: dict[str, AllowedTool] = {}
        for t in allowed_tools:
            aid = t.artifact_id or t.tool_name
            self._by_id[aid] = t

    def fetch(self, artifact_id: str, as_of: str | None = None) -> dict | None:
        tool = self._by_id.get(artifact_id)
        if tool is None:
            return None
        if as_of and tool.timestamp and tool.timestamp > as_of:
            return None
        doc = dict(tool.returns)
        if tool.subsystem:
            doc.setdefault('subsystem', tool.subsystem)
        if tool.timestamp:
            doc.setdefault('timestamp', tool.timestamp)
        doc.setdefault('id', artifact_id)
        return doc

    def search(self, query: str, artifact_type: str | None = None, as_of: str | None = None, limit: int = 10) -> list[dict]:
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
            doc.setdefault('id', aid)
            if tool.subsystem:
                doc.setdefault('subsystem', tool.subsystem)
            if tool.timestamp:
                doc.setdefault('timestamp', tool.timestamp)
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
    'type': 'object',
    'required': ['preconditions_verified', 'all_preconditions_pass', 'reasoning'],
    'properties': {
        'preconditions_verified': {
            'type': 'array',
            'items': {
                'type': 'object',
                'required': ['check', 'passed', 'facts_found'],
                'properties': {
                    'check': {'type': 'string'},
                    'passed': {'type': 'boolean'},
                    'facts_found': {'type': 'object'},
                    'evidence_artifacts': {'type': 'array', 'items': {'type': 'string'}},
                },
            },
        },
        'all_preconditions_pass': {'type': 'boolean'},
        'should_act': {'type': 'boolean'},
        'reasoning': {'type': 'string'},
        'evidence_artifacts': {'type': 'array', 'items': {'type': 'string'}},
    },
}
