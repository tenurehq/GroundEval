from __future__ import annotations

import asyncio
import importlib
import json
import logging
import sys
import time
import traceback
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from ..observe import AgentObserver, ObservedToolCall, RecordingRuntime
from .framework_observation import (
    ObservedAgent,
    ObservedError,
    ObservedEvent,
    ObservedHandoff,
    ObservedModelEvent,
    ObservedRun as RichObservedRun,
    ObservedWorkflow,
    ObservedWorkflowNode,
)

logger = logging.getLogger("groundeval.adapters.langgraph")


def _require_python_311() -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError(
            "LangGraph observation requires Python 3.11+. Earlier versions lack "
            "reliable contextvar propagation across asyncio tasks, which LangGraph "
            "and LangChain both depend on for callback/config propagation into "
            "nested and async node execution."
        )


def _load_langgraph(agent_class_path: str) -> Any:
    module_path, attr_name = agent_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    graph_obj = getattr(module, attr_name)

    if callable(graph_obj) and not isinstance(graph_obj, type):
        graph_obj = graph_obj()

    if isinstance(graph_obj, type):
        graph_obj = graph_obj()

    if hasattr(graph_obj, "compile") and callable(graph_obj.compile):
        graph_obj = graph_obj.compile()

    has_stream = hasattr(graph_obj, "stream") and callable(graph_obj.stream)
    has_astream = hasattr(graph_obj, "astream") and callable(graph_obj.astream)

    if not has_stream and not has_astream:
        raise TypeError(
            "LangGraph adapter expected a compiled graph with stream(...) or astream(...)."
        )

    return graph_obj


def _jsonish(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return value.dict()
        except Exception:
            pass
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return value.to_dict()
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _parse_jsonish(value: Any) -> Any:
    value = _jsonish(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in '[{"':
            try:
                return json.loads(stripped)
            except Exception:
                return value
    return value


def _now_str() -> str:
    return str(time.time())


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _summarize_value(value: Any, limit: int = 160) -> str:
    try:
        text = json.dumps(_jsonish(value), default=str)
    except Exception:
        text = str(value)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _extract_node_name_from_ns(ns: Any) -> str | None:
    if not ns:
        return None
    if isinstance(ns, (list, tuple)) and ns:
        last = str(ns[-1])
        if ":" in last:
            return last.split(":", 1)[0]
        return last
    return None


class _GroundEvalLangChainCallbackHandler:
    def __init__(self, collector: "_LangGraphEventCollector"):
        self.collector = collector
        self.ignore_llm = False

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        node_name = self.collector.resolve_node_name_from_callback(
            serialized=serialized,
            metadata=metadata,
            tags=tags,
            kwargs=kwargs,
        )
        self.collector.record_callback_event(
            event_type="langchain.callback.chain_start",
            node_name=node_name,
            parent_event_id=str(parent_run_id) if parent_run_id else None,
            payload={
                "run_id": str(run_id),
                "serialized": _jsonish(serialized),
                "inputs": _jsonish(inputs),
                "tags": tags or [],
                "metadata": _jsonish(metadata or {}),
            },
        )
        self.collector.note_node_start(node_name=node_name, arguments=_jsonish(inputs))

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        node_name = self.collector.resolve_node_name_from_callback(
            tags=tags,
            kwargs=kwargs,
        )
        self.collector.record_callback_event(
            event_type="langchain.callback.chain_end",
            node_name=node_name,
            parent_event_id=str(parent_run_id) if parent_run_id else None,
            payload={
                "run_id": str(run_id),
                "outputs": _jsonish(outputs),
                "tags": tags or [],
            },
        )
        self.collector.note_node_end(
            node_name=node_name, return_value=_jsonish(outputs)
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        node_name = self.collector.resolve_node_name_from_callback(
            tags=tags,
            kwargs=kwargs,
        )
        self.collector.record_callback_event(
            event_type="langchain.callback.chain_error",
            node_name=node_name,
            parent_event_id=str(parent_run_id) if parent_run_id else None,
            payload={
                "run_id": str(run_id),
                "error": repr(error),
                "tags": tags or [],
            },
        )
        self.collector.record_error(
            error_type=error.__class__.__name__,
            message=str(error),
            timestamp=_now_str(),
            executor_id=node_name,
            traceback_text="".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self.collector.record_callback_event(
            event_type="langchain.callback.llm_start",
            node_name=self.collector.resolve_node_name_from_callback(
                serialized=serialized,
                metadata=metadata,
                tags=tags,
                kwargs=kwargs,
            ),
            parent_event_id=str(parent_run_id) if parent_run_id else None,
            payload={
                "run_id": str(run_id),
                "serialized": _jsonish(serialized),
                "prompts": _jsonish(prompts),
                "tags": tags or [],
                "metadata": _jsonish(metadata or {}),
            },
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        timestamp = _now_str()
        self.collector.record_callback_event(
            event_type="langchain.callback.llm_end",
            node_name=self.collector.resolve_node_name_from_callback(
                tags=tags,
                kwargs=kwargs,
            ),
            parent_event_id=str(parent_run_id) if parent_run_id else None,
            payload={
                "run_id": str(run_id),
                "response": _jsonish(response),
                "tags": tags or [],
            },
        )
        self.collector.record_model_event(response=response, timestamp=timestamp)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        self.collector.record_callback_event(
            event_type="langchain.callback.llm_error",
            node_name=self.collector.resolve_node_name_from_callback(
                tags=tags,
                kwargs=kwargs,
            ),
            parent_event_id=str(parent_run_id) if parent_run_id else None,
            payload={
                "run_id": str(run_id),
                "error": repr(error),
                "tags": tags or [],
            },
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self.collector.record_callback_event(
            event_type="langchain.callback.chat_model_start",
            node_name=self.collector.resolve_node_name_from_callback(
                serialized=serialized,
                metadata=metadata,
                tags=tags,
                kwargs=kwargs,
            ),
            parent_event_id=str(parent_run_id) if parent_run_id else None,
            payload={
                "run_id": str(run_id),
                "serialized": _jsonish(serialized),
                "messages": _jsonish(messages),
                "tags": tags or [],
                "metadata": _jsonish(metadata or {}),
            },
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        node_name = self.collector.resolve_node_name_from_callback(
            serialized=serialized,
            metadata=metadata,
            tags=tags,
            kwargs=kwargs,
        )
        tool_name = (
            serialized.get("name")
            or serialized.get("id")
            or serialized.get("lc")
            or "tool"
        )
        parsed_inputs = (
            inputs if isinstance(inputs, dict) else _parse_jsonish(input_str)
        )
        if not isinstance(parsed_inputs, dict):
            parsed_inputs = {"input": parsed_inputs}
        self.collector.start_child_operation(
            run_id=str(run_id),
            node_name=node_name,
            tool_name=str(tool_name),
            arguments=_jsonish(parsed_inputs),
            parent_event_id=str(parent_run_id) if parent_run_id else None,
            source="child_tool",
            kind="tool",
        )
        self.collector.record_callback_event(
            event_type="langchain.callback.tool_start",
            node_name=node_name,
            parent_event_id=str(parent_run_id) if parent_run_id else None,
            payload={
                "run_id": str(run_id),
                "serialized": _jsonish(serialized),
                "input_str": input_str,
                "inputs": _jsonish(inputs),
                "tags": tags or [],
                "metadata": _jsonish(metadata or {}),
            },
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        node_name = self.collector.resolve_node_name_from_callback(
            tags=tags,
            kwargs=kwargs,
        )
        self.collector.finish_child_operation(
            run_id=str(run_id),
            node_name=node_name,
            return_value=_parse_jsonish(output),
        )
        self.collector.record_callback_event(
            event_type="langchain.callback.tool_end",
            node_name=node_name,
            parent_event_id=str(parent_run_id) if parent_run_id else None,
            payload={
                "run_id": str(run_id),
                "output": _jsonish(output),
                "tags": tags or [],
            },
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        node_name = self.collector.resolve_node_name_from_callback(
            tags=tags,
            kwargs=kwargs,
        )
        self.collector.fail_child_operation(
            run_id=str(run_id),
            node_name=node_name,
            error=error,
        )
        self.collector.record_callback_event(
            event_type="langchain.callback.tool_error",
            node_name=node_name,
            parent_event_id=str(parent_run_id) if parent_run_id else None,
            payload={
                "run_id": str(run_id),
                "error": repr(error),
                "tags": tags or [],
            },
        )

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        node_name = self.collector.resolve_node_name_from_callback(
            serialized=serialized,
            metadata=metadata,
            tags=tags,
            kwargs=kwargs,
        )
        retriever_name = (
            serialized.get("name")
            or serialized.get("id")
            or serialized.get("lc")
            or "retriever"
        )
        self.collector.start_child_operation(
            run_id=str(run_id),
            node_name=node_name,
            tool_name=str(retriever_name),
            arguments={"query": query},
            parent_event_id=str(parent_run_id) if parent_run_id else None,
            source="child_tool",
            kind="retriever",
        )
        self.collector.record_callback_event(
            event_type="langchain.callback.retriever_start",
            node_name=node_name,
            parent_event_id=str(parent_run_id) if parent_run_id else None,
            payload={
                "run_id": str(run_id),
                "serialized": _jsonish(serialized),
                "query": query,
                "tags": tags or [],
                "metadata": _jsonish(metadata or {}),
            },
        )

    def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        node_name = self.collector.resolve_node_name_from_callback(
            tags=tags,
            kwargs=kwargs,
        )
        self.collector.finish_child_operation(
            run_id=str(run_id),
            node_name=node_name,
            return_value=_jsonish(documents),
        )
        self.collector.record_callback_event(
            event_type="langchain.callback.retriever_end",
            node_name=node_name,
            parent_event_id=str(parent_run_id) if parent_run_id else None,
            payload={
                "run_id": str(run_id),
                "documents": _jsonish(documents),
                "tags": tags or [],
            },
        )

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        node_name = self.collector.resolve_node_name_from_callback(
            tags=tags,
            kwargs=kwargs,
        )
        self.collector.fail_child_operation(
            run_id=str(run_id),
            node_name=node_name,
            error=error,
        )
        self.collector.record_callback_event(
            event_type="langchain.callback.retriever_error",
            node_name=node_name,
            parent_event_id=str(parent_run_id) if parent_run_id else None,
            payload={
                "run_id": str(run_id),
                "error": repr(error),
                "tags": tags or [],
            },
        )

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: Any,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        node_name = self.collector.resolve_node_name_from_callback(
            metadata=metadata,
            tags=tags,
            kwargs=kwargs,
        )
        self.collector.record_callback_event(
            event_type="langchain.callback.custom_event",
            node_name=node_name,
            parent_event_id=None,
            payload={
                "run_id": str(run_id),
                "name": name,
                "data": _jsonish(data),
                "tags": tags or [],
                "metadata": _jsonish(metadata or {}),
            },
        )


class _LangGraphEventCollector:
    def __init__(
        self,
        *,
        run_id: str,
        agent_class: str,
        graph: Any,
        recording: RecordingRuntime,
    ):
        self.run_id = run_id
        self.agent_class = agent_class
        self.graph = graph
        self.recording = recording
        self.started_at: float | None = None
        self.completed_at: float | None = None
        self.events: list[ObservedEvent] = []
        self.tool_calls: list[ObservedToolCall] = []
        self.workflow_nodes: dict[str, ObservedWorkflowNode] = {}
        self.model_events: list[ObservedModelEvent] = []
        self.errors: list[ObservedError] = []
        self.static_handoffs: list[ObservedHandoff] = []
        self.dynamic_handoffs: list[ObservedHandoff] = []
        self._last_runtime_node_by_branch: dict[str, str] = {}
        self._runtime_handoff_keys: set[tuple[str, str, str]] = set()
        self.static_nodes: set[str] = set()
        self.static_graph_available = False
        self.subgraph_introspection_available = False
        self.callback_event_count = 0
        self.stream_event_count = 0
        self.distinct_branch_ids: set[str] = set()
        self.final_output_from_values: Any = None
        self.final_output_from_last_dict: Any = None
        self.raw_result: Any = None
        self.node_state: dict[str, dict[str, Any]] = {}
        self.child_ops_by_run_id: dict[str, dict[str, Any]] = {}
        self.child_tool_calls_by_node: dict[str, list[ObservedToolCall]] = {}
        self.callback_handler = _GroundEvalLangChainCallbackHandler(self)

    def introspect_static_graph(self) -> None:
        try:
            static_graph = self.graph.get_graph()
            self.static_graph_available = True
            static_nodes = set()
            raw_nodes = getattr(static_graph, "nodes", None)
            if isinstance(raw_nodes, dict):
                static_nodes.update(str(k) for k in raw_nodes.keys())
            elif raw_nodes is not None:
                for node in raw_nodes:
                    if hasattr(node, "id"):
                        static_nodes.add(str(node.id))
                    else:
                        static_nodes.add(str(node))
            self.static_nodes = static_nodes
            self.record_event(
                event_type="langgraph.introspection.static_nodes",
                timestamp=_now_str(),
                node_name=None,
                branch_id=None,
                parent_event_id=None,
                payload={"nodes": sorted(self.static_nodes)},
            )
            raw_edges = getattr(static_graph, "edges", None) or []
            for edge in raw_edges:
                source = str(getattr(edge, "source", ""))
                target = str(getattr(edge, "target", ""))
                if source and target:
                    self.static_handoffs.append(
                        ObservedHandoff(
                            from_executor_id=source,
                            to_executor_id=target,
                            timestamp=None,
                            payload_type="langgraph.static_edge",
                        )
                    )
        except Exception as exc:
            self.record_event(
                event_type="langgraph.introspection.unavailable",
                timestamp=_now_str(),
                node_name=None,
                branch_id=None,
                parent_event_id=None,
                payload={"error": repr(exc)},
            )

        try:
            subgraphs = self.graph.get_subgraphs()
            names: list[str] = []
            if isinstance(subgraphs, dict):
                names = sorted(str(k) for k in subgraphs.keys())
            else:
                for item in subgraphs:
                    if isinstance(item, tuple) and item:
                        names.append(str(item[0]))
                    else:
                        names.append(str(item))
                names = sorted(names)
            self.subgraph_introspection_available = True
            self.record_event(
                event_type="langgraph.introspection.subgraphs",
                timestamp=_now_str(),
                node_name=None,
                branch_id=None,
                parent_event_id=None,
                payload={"subgraphs": names},
            )
        except Exception as exc:
            self.record_event(
                event_type="langgraph.introspection.subgraphs_unavailable",
                timestamp=_now_str(),
                node_name=None,
                branch_id=None,
                parent_event_id=None,
                payload={"error": repr(exc)},
            )

    def resolve_node_name_from_callback(
        self,
        serialized: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> str | None:
        metadata = metadata or {}
        tags = tags or []
        kwargs = kwargs or {}

        for key in (
            "langgraph_node",
            "node_name",
            "checkpoint_ns",
            "langgraph_step",
            "langgraph_path",
        ):
            if key in metadata and metadata[key]:
                value = metadata[key]
                if isinstance(value, str):
                    if ":" in value:
                        return value.split(":", 1)[0]
                    if "/" in value:
                        return value.rsplit("/", 1)[-1]
                    return value

        if "name" in kwargs and kwargs["name"]:
            return str(kwargs["name"])

        if serialized:
            for key in ("name", "id"):
                if key in serialized and serialized[key]:
                    return str(serialized[key])

        return None

    def record_event(
        self,
        *,
        event_type: str,
        timestamp: str,
        node_name: str | None,
        branch_id: str | None,
        parent_event_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        self.events.append(
            ObservedEvent(
                event_type=event_type,
                timestamp=timestamp,
                node_name=node_name,
                workflow_run_id=self.run_id,
                branch_id=branch_id,
                parent_event_id=parent_event_id,
                payload=payload,
            )
        )

    def record_callback_event(
        self,
        *,
        event_type: str,
        node_name: str | None,
        parent_event_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        self.callback_event_count += 1
        self.record_event(
            event_type=event_type,
            timestamp=_now_str(),
            node_name=node_name,
            branch_id=None,
            parent_event_id=parent_event_id,
            payload=payload,
        )

    def record_error(
        self,
        *,
        error_type: str,
        message: str,
        timestamp: str,
        executor_id: str | None = None,
        traceback_text: str | None = None,
    ) -> None:
        self.errors.append(
            ObservedError(
                error_type=error_type,
                message=message,
                timestamp=timestamp,
                executor_id=executor_id,
                traceback=traceback_text,
            )
        )

    def note_node_start(self, *, node_name: str | None, arguments: Any) -> None:
        if not node_name:
            return
        state = self.node_state.setdefault(node_name, {})
        state.setdefault("entered_at", _now_str())
        if arguments is not None and "arguments" not in state:
            state["arguments"] = arguments

    def note_node_end(self, *, node_name: str | None, return_value: Any) -> None:
        if not node_name:
            return
        state = self.node_state.setdefault(node_name, {})
        state["exited_at"] = _now_str()
        if return_value is not None:
            state["return_value"] = return_value

    def note_node_from_stream(
        self,
        *,
        node_name: str | None,
        branch_id: str | None,
        arguments: Any | None,
        return_value: Any | None,
    ) -> None:
        if not node_name:
            return
        state = self.node_state.setdefault(node_name, {})
        state.setdefault("entered_at", _now_str())
        if branch_id:
            state["branch_id"] = branch_id
            self.distinct_branch_ids.add(branch_id)
        if arguments is not None and "arguments" not in state:
            state["arguments"] = arguments
        if return_value is not None:
            state["return_value"] = return_value
            state["exited_at"] = _now_str()

    def start_child_operation(
        self,
        *,
        run_id: str,
        node_name: str | None,
        tool_name: str,
        arguments: dict[str, Any],
        parent_event_id: str | None,
        source: str,
        kind: str,
    ) -> None:
        self.child_ops_by_run_id[run_id] = {
            "run_id": run_id,
            "node_name": node_name,
            "tool_name": tool_name,
            "arguments": arguments,
            "parent_event_id": parent_event_id,
            "source": source,
            "kind": kind,
            "started_at": time.time(),
        }
        if node_name:
            self.note_node_start(node_name=node_name, arguments=None)

    def finish_child_operation(
        self,
        *,
        run_id: str,
        node_name: str | None,
        return_value: Any,
    ) -> None:
        op = self.child_ops_by_run_id.pop(run_id, None)
        if op is None:
            return
        latency_ms = max(0.0, (time.time() - op["started_at"]) * 1000)
        resolved_node_name = node_name or op["node_name"]
        branch_id = self.node_state.get(resolved_node_name or "", {}).get("branch_id")
        agent_name = resolved_node_name
        agent_id = (
            f"langgraph:{branch_id or resolved_node_name}"
            if resolved_node_name
            else None
        )
        observed_call = ObservedToolCall(
            tool_name=str(op["tool_name"]),
            arguments=dict(op["arguments"]),
            return_value=return_value,
            latency_ms=latency_ms,
            agent_id=agent_id,
            agent_name=agent_name,
            node_name=resolved_node_name,
            workflow_run_id=self.run_id,
            branch_id=branch_id,
            parent_event_id=op["parent_event_id"],
        )
        self.tool_calls.append(observed_call)
        if observed_call.node_name:
            self.child_tool_calls_by_node.setdefault(
                observed_call.node_name, []
            ).append(observed_call)
        if observed_call.node_name:
            self.note_node_end(
                node_name=observed_call.node_name,
                return_value=None,
            )

    def fail_child_operation(
        self,
        *,
        run_id: str,
        node_name: str | None,
        error: BaseException,
    ) -> None:
        self.child_ops_by_run_id.pop(run_id, None)
        self.record_error(
            error_type=error.__class__.__name__,
            message=str(error),
            timestamp=_now_str(),
            executor_id=node_name,
            traceback_text="".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        )

    def record_model_event(self, *, response: Any, timestamp: str) -> None:
        model_name = None
        provider_name = None
        input_tokens = None
        output_tokens = None
        finish_reason = None
        tool_schemas_count = 0

        if hasattr(response, "llm_output") and response.llm_output:
            llm_output = _jsonish(response.llm_output)
            if isinstance(llm_output, dict):
                model_name = llm_output.get("model_name") or llm_output.get("model")
                provider_name = llm_output.get("provider")
                token_usage = llm_output.get("token_usage") or {}
                if isinstance(token_usage, dict):
                    input_tokens = _coerce_int(
                        token_usage.get("prompt_tokens")
                        or token_usage.get("input_tokens")
                    )
                    output_tokens = _coerce_int(
                        token_usage.get("completion_tokens")
                        or token_usage.get("output_tokens")
                    )

        generations = getattr(response, "generations", None)
        if generations and isinstance(generations, list) and generations:
            first_group = generations[0]
            if isinstance(first_group, list) and first_group:
                gen = first_group[0]
                msg = getattr(gen, "message", None)
                if msg is not None:
                    response_metadata = getattr(msg, "response_metadata", None) or {}
                    if isinstance(response_metadata, dict):
                        finish_reason = response_metadata.get("finish_reason")
                        model_name = model_name or response_metadata.get("model_name")
                    usage_metadata = getattr(msg, "usage_metadata", None) or {}
                    if isinstance(usage_metadata, dict):
                        input_tokens = input_tokens or _coerce_int(
                            usage_metadata.get("input_tokens")
                        )
                        output_tokens = output_tokens or _coerce_int(
                            usage_metadata.get("output_tokens")
                        )
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    if isinstance(tool_calls, list):
                        tool_schemas_count = len(tool_calls)

        self.model_events.append(
            ObservedModelEvent(
                event_type="model.call.completed",
                timestamp=timestamp,
                model_name=model_name,
                provider_name=provider_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                tool_schemas_count=tool_schemas_count,
            )
        )

    def record_runtime_transition(
        self,
        *,
        node_name: str | None,
        branch_id: str | None,
        timestamp: str,
    ) -> None:
        if not node_name or node_name in {"__start__", "__end__"}:
            return

        branch_key = branch_id or "__root__"
        previous = self._last_runtime_node_by_branch.get(branch_key)
        self._last_runtime_node_by_branch[branch_key] = node_name

        if not previous or previous == node_name:
            return

        static_pairs = {
            (handoff.from_executor_id, handoff.to_executor_id)
            for handoff in self.static_handoffs
        }
        if static_pairs and (previous, node_name) not in static_pairs:
            return

        key = (branch_key, previous, node_name)
        if key in self._runtime_handoff_keys:
            return

        self._runtime_handoff_keys.add(key)
        self.dynamic_handoffs.append(
            ObservedHandoff(
                from_executor_id=previous,
                to_executor_id=node_name,
                timestamp=timestamp,
                payload_type="langgraph.runtime_transition",
            )
        )

    def process_stream_chunk(self, chunk: Any) -> None:
        timestamp = _now_str()
        self.stream_event_count += 1

        if isinstance(chunk, dict) and "type" in chunk:
            mode = str(chunk.get("type"))
            namespace = chunk.get("ns") or ()
            branch_id = "/".join(str(item) for item in namespace) if namespace else None
            node_name = _extract_node_name_from_ns(namespace)
            data = _jsonish(chunk.get("data"))
            self._process_normalized_stream_chunk(
                mode=mode,
                data=data,
                node_name=node_name,
                branch_id=branch_id,
                timestamp=timestamp,
            )
            return

        if isinstance(chunk, tuple) and len(chunk) == 3:
            namespace, mode, data = chunk
            branch_id = (
                "/".join(str(item) for item in namespace)
                if isinstance(namespace, (list, tuple)) and namespace
                else str(namespace)
                if namespace
                else None
            )
            node_name = _extract_node_name_from_ns(namespace)
            self._process_normalized_stream_chunk(
                mode=str(mode),
                data=_jsonish(data),
                node_name=node_name,
                branch_id=branch_id,
                timestamp=timestamp,
            )
            return

        if isinstance(chunk, tuple) and len(chunk) == 2:
            first, data = chunk
            if isinstance(first, (list, tuple)):
                namespace = first
                branch_id = (
                    "/".join(str(item) for item in namespace) if namespace else None
                )
                node_name = _extract_node_name_from_ns(namespace)
                mode = "updates"
            else:
                branch_id = None
                node_name = None
                mode = str(first)

            self._process_normalized_stream_chunk(
                mode=mode,
                data=_jsonish(data),
                node_name=node_name,
                branch_id=branch_id,
                timestamp=timestamp,
            )
            return

        self.record_event(
            event_type="langgraph.stream.unknown",
            timestamp=timestamp,
            node_name=None,
            branch_id=None,
            parent_event_id=None,
            payload={"raw": _jsonish(chunk)},
        )

    def _process_normalized_stream_chunk(
        self,
        *,
        mode: str,
        data: Any,
        node_name: str | None,
        branch_id: str | None,
        timestamp: str,
    ) -> None:
        if branch_id:
            self.distinct_branch_ids.add(branch_id)

        self.record_event(
            event_type=f"langgraph.stream.{mode}",
            timestamp=timestamp,
            node_name=node_name,
            branch_id=branch_id,
            parent_event_id=None,
            payload={"data": data},
        )
        self._extract_node_data_from_stream(
            mode=mode,
            data=data,
            node_name=node_name,
            branch_id=branch_id,
        )

        if mode == "updates" and isinstance(data, dict) and len(data) == 1:
            executed_node = str(next(iter(data)))
            self.record_runtime_transition(
                node_name=executed_node,
                branch_id=branch_id,
                timestamp=timestamp,
            )
        elif mode == "debug" and isinstance(data, dict):
            executed_node = (
                data.get("name")
                or data.get("node")
                or data.get("task")
                or data.get("task_name")
                or node_name
            )
            self.record_runtime_transition(
                node_name=str(executed_node) if executed_node else None,
                branch_id=branch_id,
                timestamp=timestamp,
            )

        if mode == "values" and isinstance(data, dict):
            self.final_output_from_values = data
            self.final_output_from_last_dict = data
        elif isinstance(data, dict):
            self.final_output_from_last_dict = data

    def _extract_node_data_from_stream(
        self,
        *,
        mode: str,
        data: Any,
        node_name: str | None,
        branch_id: str | None,
    ) -> None:
        if mode == "updates" and isinstance(data, dict):
            for candidate_node, candidate_output in data.items():
                self.note_node_from_stream(
                    node_name=str(candidate_node),
                    branch_id=branch_id,
                    arguments=None,
                    return_value=_jsonish(candidate_output),
                )
            return

        if mode == "debug" and isinstance(data, dict):
            payload = data
            candidate_node = (
                payload.get("name")
                or payload.get("node")
                or payload.get("task")
                or payload.get("task_name")
                or node_name
            )
            candidate_input = (
                payload.get("input")
                or payload.get("inputs")
                or payload.get("state")
                or payload.get("task_input")
            )
            candidate_output = (
                payload.get("result")
                or payload.get("output")
                or payload.get("state_update")
                or payload.get("task_result")
            )
            self.note_node_from_stream(
                node_name=str(candidate_node) if candidate_node else None,
                branch_id=branch_id,
                arguments=_jsonish(candidate_input)
                if candidate_input is not None
                else None,
                return_value=_jsonish(candidate_output)
                if candidate_output is not None
                else None,
            )
            return

        if node_name and isinstance(data, dict):
            self.note_node_from_stream(
                node_name=node_name,
                branch_id=branch_id,
                arguments=None,
                return_value=data,
            )

    def _node_tool_name(self, node_name: str) -> str:
        return node_name

    def _build_workflow(self, *, final_output: Any) -> ObservedWorkflow:
        all_node_names = set(self.static_nodes) | set(self.node_state.keys())
        for node_name in sorted(all_node_names):
            state = self.node_state.get(node_name, {})
            child_calls = self.child_tool_calls_by_node.get(node_name, [])
            entered_at = state.get("entered_at")
            exited_at = state.get("exited_at")
            arguments = state.get("arguments", {})
            return_value = state.get("return_value")

            if node_name not in self.node_state:
                node_type = "langgraph.node.not_observed_in_run"
            elif child_calls:
                node_type = "langgraph.node.observed_with_children"
            else:
                node_type = "langgraph.node.observed_operation"
                branch_id = state.get("branch_id")
                observed_call = ObservedToolCall(
                    tool_name=self._node_tool_name(node_name),
                    arguments=arguments if isinstance(arguments, dict) else {},
                    return_value=return_value,
                    latency_ms=0.0,
                    agent_id=f"langgraph:{branch_id or node_name}",
                    agent_name=node_name,
                    node_name=node_name,
                    workflow_run_id=self.run_id,
                    branch_id=branch_id,
                    parent_event_id=None,
                )
                self.tool_calls.append(observed_call)

            self.workflow_nodes[node_name] = ObservedWorkflowNode(
                node_id=node_name,
                node_type=node_type,
                entered_at=entered_at,
                exited_at=exited_at,
                agent_name=node_name if node_name in self.node_state else None,
            )

        all_handoffs = self.static_handoffs + self.dynamic_handoffs

        return ObservedWorkflow(
            workflow_id=self.run_id,
            workflow_name="LangGraph workflow trace",
            workflow_description=None,
            node_count=len(self.workflow_nodes),
            nodes=list(self.workflow_nodes.values()),
            handoff_count=len(all_handoffs),
            handoffs=all_handoffs,
            branch_count=len(self.distinct_branch_ids),
        )

    def _final_output(self) -> Any:
        if isinstance(self.final_output_from_values, dict):
            return self.final_output_from_values
        if isinstance(self.final_output_from_last_dict, dict):
            return self.final_output_from_last_dict
        if isinstance(self.raw_result, dict):
            return self.raw_result
        return {"raw_output": str(self.raw_result)[:1000]}

    async def _execute_async(self, agent: Any) -> Any:
        inputs = {}
        config = {
            "callbacks": [self.callback_handler],
            "tags": ["groundeval", "groundeval:langgraph"],
            "metadata": {
                "groundeval_observed_run_id": self.run_id,
                "groundeval_adapter": "langgraph",
            },
            "configurable": {
                "groundeval_observed_run_id": self.run_id,
                "groundeval_adapter": "langgraph",
            },
        }

        max_steps = getattr(agent, "_groundeval_max_steps", None)
        if isinstance(max_steps, int) and max_steps > 0:
            config["recursion_limit"] = max_steps

        if hasattr(agent, "astream") and callable(agent.astream):
            async_stream = cast(
                AsyncIterator[Any],
                agent.astream(
                    inputs,
                    config=config,
                    stream_mode=["values", "updates", "debug"],
                    subgraphs=True,
                ),
            )
            async for chunk in async_stream:
                self.process_stream_chunk(chunk)
        elif hasattr(agent, "stream") and callable(agent.stream):
            sync_stream = cast(
                Iterator[Any],
                agent.stream(
                    inputs,
                    config=config,
                    stream_mode=["values", "updates", "debug"],
                    subgraphs=True,
                ),
            )
            for chunk in sync_stream:
                self.process_stream_chunk(chunk)
        else:
            raise TypeError(
                "LangGraph adapter expected a compiled graph with stream(...) or astream(...)."
            )

        return self.final_output_from_values or self.final_output_from_last_dict

    def execute(self, agent: Any) -> Any:
        self.started_at = time.time()
        self.introspect_static_graph()
        try:
            self.raw_result = asyncio.run(self._execute_async(agent))
            self.completed_at = time.time()
            return self.raw_result
        except Exception as exc:
            self.completed_at = time.time()
            self.raw_result = None
            self.record_error(
                error_type=exc.__class__.__name__,
                message=str(exc),
                timestamp=_now_str(),
                executor_id=None,
                traceback_text="".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            )
            raise

    def to_observed_run(self, result: Any) -> RichObservedRun:
        if result is not None:
            self.raw_result = result

        final_output = self._final_output()
        workflow = self._build_workflow(final_output=final_output)

        capabilities = {
            "langgraph_stream": self.stream_event_count > 0,
            "langchain_callbacks": self.callback_event_count > 0,
            "tool_calls": bool(self.tool_calls),
            "agent_turns": bool(self.workflow_nodes),
            "workflow_nodes": bool(self.workflow_nodes),
            "handoffs": bool(self.dynamic_handoffs),
            "static_graph_edges": bool(self.static_handoffs),
            "approvals": False,
            "checkpoints": False,
            "context_injection": False,
            "model_calls": bool(self.model_events),
            "static_graph_introspection": self.static_graph_available,
            "subgraph_introspection": self.subgraph_introspection_available,
            "subgraph_namespaces": bool(self.distinct_branch_ids),
        }

        started_at = str(self.started_at) if self.started_at is not None else None
        completed_at = str(self.completed_at) if self.completed_at is not None else None
        total_latency_ms = 0.0
        if self.started_at is not None and self.completed_at is not None:
            total_latency_ms = (self.completed_at - self.started_at) * 1000

        return RichObservedRun(
            run_id=self.run_id,
            framework="langgraph",
            agent_class=self.agent_class,
            started_at=started_at,
            completed_at=completed_at,
            total_latency_ms=total_latency_ms,
            tool_calls=self.tool_calls,
            events=self.events,
            agents=[
                ObservedAgent(
                    agent_id=f"langgraph:{node.node_id}",
                    agent_name=node.node_id,
                    role=node.node_id,
                    tool_call_count=sum(
                        1 for tc in self.tool_calls if tc.agent_name == node.node_id
                    ),
                )
                for node in self.workflow_nodes.values()
                if node.agent_name
            ],
            workflow=workflow,
            model_events=self.model_events,
            final_output=final_output,
            errors=self.errors,
            capabilities=capabilities,
        )


class LangGraphObserver(AgentObserver):
    def load_agent(self, class_path: str) -> Any:
        _require_python_311()
        return _load_langgraph(class_path)

    def instrument_agent(
        self,
        agent: Any,
        recording: RecordingRuntime,
    ) -> Any:
        _require_python_311()
        run_id = f"langgraph_observed_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        collector = _LangGraphEventCollector(
            run_id=run_id,
            agent_class=f"{agent.__class__.__module__}.{agent.__class__.__name__}",
            graph=agent,
            recording=recording,
        )
        agent._groundeval_langgraph_collector = collector
        agent._groundeval_recording = recording
        return agent

    def execute_agent(self, agent: Any) -> Any:
        collector = agent._groundeval_langgraph_collector
        result = None
        try:
            result = collector.execute(agent)
            return result
        finally:
            observed_run = collector.to_observed_run(result)
            agent._groundeval_framework_observed_run = observed_run
            recording = getattr(agent, "_groundeval_recording", None)
            if recording is not None:
                for tc in observed_run.tool_calls:
                    recording.record(
                        tool_name=tc.tool_name,
                        arguments=tc.arguments,
                        return_value=tc.return_value,
                        latency_ms=tc.latency_ms,
                        agent_id=tc.agent_id,
                        agent_name=tc.agent_name,
                        node_name=tc.node_name,
                        workflow_run_id=tc.workflow_run_id,
                        branch_id=tc.branch_id,
                        parent_event_id=tc.parent_event_id,
                    )

    def set_max_steps(self, agent: Any, max_steps: int) -> None:
        if max_steps <= 0:
            raise ValueError("LangGraph max_steps must be greater than zero.")
        agent._groundeval_max_steps = int(max_steps)


def generate_langgraph_report(run: RichObservedRun) -> str:
    workflow_nodes = run.workflow.nodes if run.workflow else []
    handoffs = run.workflow.handoffs if run.workflow else []

    source_by_call: dict[str, str] = {}
    observed_with_children = {
        node.node_id
        for node in workflow_nodes
        if node.node_type == "langgraph.node.observed_with_children"
    }
    for tc in run.tool_calls:
        source_by_call[
            f"{tc.tool_name}|{tc.node_name}|{json.dumps(_jsonish(tc.arguments), default=str)}"
        ] = "child_tool" if tc.node_name in observed_with_children else "node"

    lines = [
        "# GroundEval LangGraph Observation Report",
        "",
        "## Summary",
        f"- Run ID: `{run.run_id}`",
        f"- Framework: {run.framework}",
        f"- Agent class: {run.agent_class}",
        f"- Total latency: {run.total_latency_ms:.0f}ms",
        f"- Tool calls recorded (deterministic runtime operations): {len(run.tool_calls)}",
        f"- Agent inventory: {len(run.agents)}",
        f"- Nodes observed (total, including non-tool-call nodes): {len(workflow_nodes)}",
        f"- Model events: {len(run.model_events)}",
        f"- Workflow nodes: {len(workflow_nodes)}",
        "",
        "## Capabilities",
        "",
        "| Capability | Observed |",
        "|---|---|",
    ]

    for key, value in sorted(run.capabilities.items()):
        lines.append(f"| {key} | {'Yes' if value else 'No'} |")

    lines.extend([
        "",
        "## Workflow Nodes",
        "",
        "| Node | Status | Entered | Exited |",
        "|---|---|---|---|",
    ])
    for node in workflow_nodes:
        lines.append(
            f"| `{node.node_id}` | {node.node_type or ''} | {node.entered_at or ''} | {node.exited_at or ''} |"
        )

    lines.extend([
        "",
        "## Observed Operations",
        "",
        "| Tool/operation name | Node | Source | Arguments | Return summary | Latency |",
        "|---|---|---|---|---|---:|",
    ])
    for tc in run.tool_calls:
        key = f"{tc.tool_name}|{tc.node_name}|{json.dumps(_jsonish(tc.arguments), default=str)}"
        source = source_by_call.get(key, "node")
        lines.append(
            f"| `{tc.tool_name}` | `{tc.node_name or ''}` | {source} | "
            f"`{_summarize_value(tc.arguments)}` | `{_summarize_value(tc.return_value)}` | "
            f"{tc.latency_ms:.0f} |"
        )

    lines.extend([
        "",
        "## Static Handoffs",
        "",
        "| From | To | Payload type |",
        "|---|---|---|",
    ])
    for h in handoffs:
        lines.append(
            f"| `{h.from_executor_id}` | `{h.to_executor_id}` | {h.payload_type or ''} |"
        )

    lines.extend([
        "",
        "## Model Events",
        "",
        "| Model | Provider | Input tokens | Output tokens | Finish reason |",
        "|---|---|---:|---:|---|",
    ])
    for m in run.model_events:
        lines.append(
            f"| {m.model_name or ''} | {m.provider_name or ''} | "
            f"{m.input_tokens if m.input_tokens is not None else ''} | "
            f"{m.output_tokens if m.output_tokens is not None else ''} | "
            f"{m.finish_reason or ''} |"
        )

    lines.extend([
        "",
        "## Raw Framework Events",
        "",
        "| Event type | Node | Branch | Timestamp |",
        "|---|---|---|---|",
    ])
    for e in run.events:
        lines.append(
            f"| {e.event_type} | {e.node_name or ''} | {e.branch_id or ''} | {e.timestamp or ''} |"
        )

    lines.extend([
        "",
        "## Errors",
        "",
        "| Error type | Message | Timestamp |",
        "|---|---|---|",
    ])
    for e in run.errors:
        lines.append(f"| {e.error_type} | {e.message} | {e.timestamp or ''} |")

    return "\n".join(lines)
