"""
groundeval/adapters.py
===========================
Built-in implementations of CorpusAdapter and AccessPolicy.

FileCorpusAdapter  — artifacts live in a flat directory of JSON files.
                     Filename without extension = artifact ID.
NullCorpusAdapter  — context-injection mode. No retrieval; the framework
                     pre-loads context into the agent's prompt instead.
YamlAccessPolicy   — roles and subsystem access declared in config.yaml.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from .core import (
    CorpusAdapter,
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
        subsystem_map: dict[str, str] | None = None,
    ):
        self._root = Path(root_dir)
        self._subsystem_map = subsystem_map or {}
        self._cache: dict[str, dict] = {}
        self._index: dict[str, Path] = {}
        self._build_index()

    def _build_index(self) -> None:
        for p in self._root.rglob("*.json"):
            try:
                doc = json.loads(p.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._index[p.stem] = p
                continue

            if isinstance(doc, list):
                for i, item in enumerate(doc):
                    if isinstance(item, dict):
                        aid = item.get("id", item.get("_id", ""))
                        if aid:
                            if aid in self._index:
                                import warnings

                                warnings.warn(
                                    f"FileCorpusAdapter: duplicate artifact ID "
                                    f"'{aid}' found in {p} at index {i}. "
                                    f"Previous entry will be overwritten.",
                                    stacklevel=2,
                                )
                            self._index[aid] = (p, i)
            elif isinstance(doc, dict):
                self._index[p.stem] = p
            else:
                self._index[p.stem] = p

    def _load(self, artifact_id: str) -> dict | None:
        if artifact_id in self._cache:
            return self._cache[artifact_id]
        entry = self._index.get(artifact_id)
        if not entry:
            return None

        if isinstance(entry, tuple):
            # Array file entry: (path, index)
            path, idx = entry
            doc = json.loads(path.read_text())
            if isinstance(doc, list) and 0 <= idx < len(doc):
                doc = doc[idx]
            else:
                return None
        else:
            doc = json.loads(entry.read_text())

        if "subsystem" not in doc and isinstance(entry, tuple):
            # For array files, subsystem comes from the object itself
            pass
        elif "subsystem" not in doc and entry.parent != self._root:
            doc["subsystem"] = entry.parent.name

        self._cache[artifact_id] = doc
        return doc

    def fetch(self, artifact_id: str, as_of: str | None = None) -> dict | None:
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
        artifact_type: str | None = None,
        as_of: str | None = None,
        limit: int = 10,
    ) -> list[dict]:

        if limit <= 0:
            return []

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

    def timestamp_of(self, artifact_id: str) -> str | None:
        doc = self._load(artifact_id)
        if not doc:
            return None
        return doc.get("timestamp") or doc.get("created_at") or doc.get("date")

    def subsystem_of(self, artifact_id: str) -> str | None:
        if artifact_id in self._subsystem_map:
            return self._subsystem_map[artifact_id]
        doc = self._load(artifact_id)
        return doc.get("subsystem") if doc else None

    def list_ids(self, subsystem: str | None = None) -> list[str]:
        if not subsystem:
            return list(self._index.keys())
        return [aid for aid in self._index if self.subsystem_of(aid) == subsystem]


class NullCorpusAdapter:
    """
    Used when artifacts are pre-loaded into the agent's context window.
    All retrieval methods return None / [].
    The framework still enforces gate logic for citation scoring.
    """

    def fetch(self, artifact_id: str, as_of: str | None = None) -> None:
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
        self._actors: dict[str, str] = config.get("actors", {})
        self._roles: dict[str, dict] = config.get("roles", {})

    @classmethod
    def from_file(cls, path: str | Path) -> YamlAccessPolicy:
        with open(path) as f:
            data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Expected YAML mapping, got {type(data).__name__}")
            return cls(data)

    def subsystems_for_role(self, role: str) -> set[str]:
        role_cfg = self._roles.get(role, {})
        return set(role_cfg.get("subsystems", []))

    def role_for_actor(self, actor_id: str) -> str | None:
        return self._actors.get(actor_id)

    def visible_artifacts(
        self,
        actor_id: str,
        all_artifact_ids: list[str],
        as_of: str | None = None,
        corpus: CorpusAdapter | None = None,
    ) -> set[str]:
        role = self.role_for_actor(actor_id)
        if not role:
            return set()
        accessible = self.subsystems_for_role(role)
        if corpus is None:
            return set()
        visible = set()
        for aid in all_artifact_ids:
            subsystem = corpus.subsystem_of(aid)
            if subsystem in accessible or subsystem is None:
                visible.add(aid)
        return visible


class InMemoryCorpusAdapter:
    """
    Corpus adapter that serves artifacts from an in-memory list.

    Used for distractor mode where seed + generated artifacts are
    held in memory rather than on disk.
    """

    def __init__(self, artifacts: list[dict[str, Any]]):
        self._by_id: dict[str, dict] = {}
        for a in artifacts:
            aid = a.get("id", a.get("_id", ""))
            if aid:
                self._by_id[aid] = a

    def fetch(self, artifact_id: str, as_of: str | None = None) -> dict | None:
        doc = self._by_id.get(artifact_id)
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
        artifact_type: str | None = None,
        as_of: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        if limit <= 0:
            return []

        import re

        pattern = re.compile(re.escape(query), re.IGNORECASE)
        results = []
        for doc in self._by_id.values():
            if artifact_type and doc.get("subsystem") != artifact_type:
                continue
            if as_of:
                ts = (
                    doc.get("timestamp") or doc.get("created_at") or doc.get("date", "")
                )
                if ts and ts > as_of:
                    continue
            if pattern.search(json.dumps(doc)):
                results.append(doc)
            if len(results) >= limit:
                break
        return results

    def timestamp_of(self, artifact_id: str) -> str | None:
        doc = self._by_id.get(artifact_id)
        if not doc:
            return None
        return doc.get("timestamp") or doc.get("created_at") or doc.get("date")

    def subsystem_of(self, artifact_id: str) -> str | None:
        doc = self._by_id.get(artifact_id)
        return doc.get("subsystem") if doc else None

    def list_ids(self, subsystem: str | None = None) -> list[str]:
        if not subsystem:
            return list(self._by_id.keys())
        return [
            aid for aid, doc in self._by_id.items() if doc.get("subsystem") == subsystem
        ]
