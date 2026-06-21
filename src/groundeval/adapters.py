"""
groundeval/adapters.py
===========================
Built-in implementations of CorpusAdapter and AccessPolicy.

FileCorpusAdapter  — artifacts live in a flat directory of JSON files.
                     Filename without extension = artifact ID.
NullCorpusAdapter  — context-injection mode. No retrieval; the framework
                     pre-loads context into the agent's prompt instead.
YamlAccessPolicy   — roles and subsystem access declared in config.yaml.
EventLogPolicy     — derives visibility from the event log itself
                     (actor appears in event.actors -> can see that event's artifacts).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

from .core import (
    CorpusAdapter,
    LogEvent,
)


class FileCorpusAdapter:
    """
    Artifacts are JSON files in `root_dir`.

    Directory layout (flat or one-level nested by subsystem):
        root_dir/
            email-001.json          <- artifact_id = "email-001"
            jira/
                TICKET-42.json      <- artifact_id = "TICKET-42"

    Each JSON file must contain at least:
        {
          "id": "...",
          "timestamp": "2026-01-15T09:30:00",
          "subsystem": "email",     <- optional; inferred from directory name
          ...                       <- any other fields passed through as-is
        }

    subsystem_map overrides directory-based subsystem inference:
        {"email-001": "email", "TICKET-42": "jira"}
    """

    def __init__(
        self,
        root_dir: str | Path,
        subsystem_map: Optional[Dict[str, str]] = None,
    ):
        self._root = Path(root_dir)
        self._subsystem_map = subsystem_map or {}
        self._cache: Dict[str, dict] = {}
        self._index: Dict[str, Path] = {}
        self._build_index()

    def _build_index(self) -> None:
        for p in self._root.rglob("*.json"):
            artifact_id = p.stem
            self._index[artifact_id] = p

    def _load(self, artifact_id: str) -> Optional[dict]:
        if artifact_id in self._cache:
            return self._cache[artifact_id]
        path = self._index.get(artifact_id)
        if not path:
            return None
        doc = json.loads(path.read_text())
        if "subsystem" not in doc and path.parent != self._root:
            doc["subsystem"] = path.parent.name
        self._cache[artifact_id] = doc
        return doc

    def fetch(self, artifact_id: str, as_of: Optional[str] = None) -> Optional[dict]:
        doc = self._load(artifact_id)
        if doc is None:
            return None
        if as_of:
            ts = doc.get("timestamp") or doc.get("created_at") or doc.get("date", "")
            if ts and ts > as_of:
                return None
        return doc

    def search(
        self,
        query: str,
        artifact_type: Optional[str] = None,
        as_of: Optional[str] = None,
        limit: int = 10,
    ) -> List[dict]:
        results = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        for artifact_id, _path in self._index.items():
            doc = self._load(artifact_id)
            if doc is None:
                continue
            if artifact_type and doc.get("subsystem") != artifact_type:
                continue
            if as_of:
                ts = (
                    doc.get("timestamp") or doc.get("created_at") or doc.get("date", "")
                )
                if ts and ts > as_of:
                    continue
            text = json.dumps(doc)
            if pattern.search(text):
                results.append(doc)
            if len(results) >= limit:
                break
        return results

    def timestamp_of(self, artifact_id: str) -> Optional[str]:
        doc = self._load(artifact_id)
        if not doc:
            return None
        return doc.get("timestamp") or doc.get("created_at") or doc.get("date")

    def subsystem_of(self, artifact_id: str) -> Optional[str]:
        if artifact_id in self._subsystem_map:
            return self._subsystem_map[artifact_id]
        doc = self._load(artifact_id)
        return doc.get("subsystem") if doc else None

    def list_ids(self, subsystem: Optional[str] = None) -> List[str]:
        if not subsystem:
            return list(self._index.keys())
        return [aid for aid in self._index if self.subsystem_of(aid) == subsystem]


class NullCorpusAdapter:
    """
    Used when artifacts are pre-loaded into the agent's context window.
    All retrieval methods return None / [].
    The framework still enforces gate logic for citation scoring.
    """

    def fetch(self, artifact_id: str, as_of: Optional[str] = None) -> None:
        return None

    def search(self, query: str, artifact_type=None, as_of=None, limit=10) -> list:
        return []

    def timestamp_of(self, artifact_id: str) -> None:
        return None

    def subsystem_of(self, artifact_id: str) -> None:
        return None

    def list_ids(self, subsystem=None) -> list:
        return []


class YamlAccessPolicy:
    """
    Role-based access policy declared in config.yaml.

    Expected config structure:

        actors:
          alice: engineer
          bob: sales
          carol: admin

        roles:
          engineer:
            subsystems: [jira, git, slack, confluence, email]
          sales:
            subsystems: [salesforce, email, slack]
          admin:
            subsystems: [jira, git, slack, confluence, email, salesforce, zendesk]

    Visibility is subsystem-based: if the artifact's subsystem is in the
    actor's role subsystem list, the actor can see it.

    For finer-grained control (direct involvement only), use EventLogPolicy
    or subclass this and override visible_artifacts().
    """

    def __init__(self, config: dict):
        self._actors: Dict[str, str] = config.get("actors", {})
        self._roles: Dict[str, Dict] = config.get("roles", {})

    @classmethod
    def from_file(cls, path: str | Path) -> "YamlAccessPolicy":
        with open(path) as f:
            data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Expected YAML mapping, got {type(data).__name__}")
            return cls(data)

    def subsystems_for_role(self, role: str) -> Set[str]:
        role_cfg = self._roles.get(role, {})
        return set(role_cfg.get("subsystems", []))

    def role_for_actor(self, actor_id: str) -> Optional[str]:
        return self._actors.get(actor_id)

    def visible_artifacts(
        self,
        actor_id: str,
        all_artifact_ids: List[str],
        as_of: Optional[str] = None,
        corpus: Optional[CorpusAdapter] = None,
    ) -> Set[str]:
        role = self.role_for_actor(actor_id)
        if not role:
            return set()
        accessible = self.subsystems_for_role(role)
        if corpus is None:
            return set(all_artifact_ids)
        visible = set()
        for aid in all_artifact_ids:
            subsystem = corpus.subsystem_of(aid)
            if subsystem in accessible or subsystem is None:
                visible.add(aid)
        return visible


class EventLogPolicy:
    """
    Derives actor visibility from event log participation.

    An actor can see artifact X if:
      1. The actor appears in event.actors for the event that created X, OR
      2. The event type is in the actor's broadcast_event_types (role config)

    This is the richer model that simulation harnesses use internally.

    config structure (in addition to YamlAccessPolicy fields):

        roles:
          engineer:
            subsystems: [jira, git, slack, confluence]
            broadcast_event_types: [incident_opened, incident_resolved]
    """

    def __init__(self, config: dict, events: List[LogEvent]):
        self._base = YamlAccessPolicy(config)
        self._events = events
        self._roles: Dict[str, Dict] = config.get("roles", {})
        self._artifact_actors: Dict[str, Set[str]] = {}
        self._artifact_event_type: Dict[str, str] = {}
        self._build_index()

    def _build_index(self) -> None:
        for event in self._events:
            for _key, val in event.artifact_ids.items():
                ids = val if isinstance(val, list) else [val]
                for aid in ids:
                    if not aid:
                        continue
                    if aid not in self._artifact_actors:
                        self._artifact_actors[aid] = set()
                    self._artifact_actors[aid].update(event.actors)
                    self._artifact_event_type[aid] = event.type

    @classmethod
    def from_file(
        cls, config_path: str | Path, events: List[LogEvent]
    ) -> "EventLogPolicy":
        with open(config_path) as f:
            data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Expected YAML mapping, got {type(data).__name__}")
            return cls(data, events)

    def subsystems_for_role(self, role: str) -> Set[str]:
        return self._base.subsystems_for_role(role)

    def role_for_actor(self, actor_id: str) -> Optional[str]:
        return self._base.role_for_actor(actor_id)

    def visible_artifacts(
        self,
        actor_id: str,
        all_artifact_ids: List[str],
        as_of: Optional[str] = None,
        corpus: Optional[CorpusAdapter] = None,
    ) -> Set[str]:
        role = self.role_for_actor(actor_id)
        if not role:
            return set()

        accessible_subsystems = self.subsystems_for_role(role)
        broadcast_types = set(
            self._roles.get(role, {}).get("broadcast_event_types", [])
        )

        visible = set()
        for aid in all_artifact_ids:
            if as_of and corpus:
                ts = corpus.timestamp_of(aid)
                if ts and ts > as_of:
                    continue

            if actor_id in self._artifact_actors.get(aid, set()):
                visible.add(aid)
                continue

            event_type = self._artifact_event_type.get(aid, "")
            if event_type in broadcast_types:
                visible.add(aid)
                continue

            subsystem = corpus.subsystem_of(aid) if corpus else None
            if subsystem and subsystem not in accessible_subsystems:
                continue
            visible.add(aid)

        return visible
