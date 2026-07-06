import json


import pytest
from reportlab.pdfgen import canvas as rl_canvas

from groundeval.diagram import (
    _build_lanes_and_steps,
    _deserialize_rich_run,
    _lane_label,
    _resolve_lane_id,
    _safe_json,
    _summarize_arguments,
    _summarize_return_value,
    _truncate,
    _wrap_text,
    render_observe_diagram_pdf,
    write_observe_diagram_pdf,
)
from groundeval.observe import ObservedRun, ObservedToolCall


def _make_tool_call(
    tool_name="fetch_customer",
    agent_id=None,
    agent_name=None,
    arguments=None,
    return_value=None,
    latency_ms=1.0,
):
    return ObservedToolCall(
        tool_name=tool_name,
        arguments={} if arguments is None else arguments,
        return_value=return_value,
        latency_ms=latency_ms,
        agent_id=agent_id,
        agent_name=agent_name,
    )


def _make_run(
    tool_calls=None,
    final_answer=None,
    framework="custom",
    framework_extra=None,
    run_id="run-1",
    agent_class="pkg.Agent",
    total_latency_ms=123.0,
):
    return ObservedRun(
        run_id=run_id,
        framework=framework,
        agent_class=agent_class,
        tool_calls=[] if tool_calls is None else tool_calls,
        final_answer=final_answer,
        framework_extra=framework_extra,
        total_latency_ms=total_latency_ms,
    )


def test_safe_json_and_truncate_helpers_cover_common_cases():
    assert _safe_json(None, fallback="fallback") == "fallback"
    assert _safe_json("plain") == "plain"
    assert _safe_json({"a": 1}) == json.dumps({"a": 1})
    assert _truncate("short", 10) == "short"
    assert _truncate("line1 line2", 20) == "line1 line2"
    assert _truncate("abcdefghijklmnopqrstuvwxyz", 8) == "abcde..."


def test_summarize_helpers_cover_dict_list_and_scalars():
    args = {
        "customer_id": 123,
        "query": "find open ticket",
        "verbose": True,
        "ignored": "fourth item",
    }
    summary = _summarize_arguments(args)
    assert "customer_id=123" in summary
    assert "query=" in summary
    assert "verbose=true" in summary.lower()
    assert "ignored" not in summary

    assert _summarize_return_value({"summary": "done"}) == "done"
    assert _summarize_return_value([]) == "empty result"
    assert _summarize_return_value([{"id": 1}]).startswith("1 result:")
    assert _summarize_return_value([1, 2, 3]) == "3 results"
    assert _summarize_return_value("plain text") == "plain text"


def test_wrap_text_splits_long_content_into_multiple_lines():
    lines = _wrap_text(
        "one two three four five six", max_width=40, font_name="Helvetica", font_size=8
    )
    assert len(lines) >= 2
    assert " ".join(lines).replace("  ", " ") == "one two three four five six"


def test_lane_label_and_resolve_lane_id_prefer_expected_values():
    assert _lane_label("agent-1", "Planner") == "Planner"
    assert _lane_label("agent-1", "agent-1") == "agent-1"
    assert _lane_label(None, None) == "Agent"

    known_agents = {"agent-1": "Planner"}
    tc_by_id = _make_tool_call(agent_id="agent-1", agent_name="Planner")
    tc_by_name = _make_tool_call(agent_name="Planner")
    tc_unknown = _make_tool_call()

    assert _resolve_lane_id(tc_by_id, known_agents) == "agent-1"
    assert _resolve_lane_id(tc_by_name, known_agents) == "agent-1"
    assert _resolve_lane_id(tc_unknown, known_agents) == "system"


def test_deserialize_rich_run_returns_none_without_framework_extra():
    observed = _make_run(tool_calls=[])
    assert _deserialize_rich_run(observed) is None


def test_deserialize_rich_run_returns_none_when_adapter_import_fails(monkeypatch):
    observed = _make_run(tool_calls=[], framework_extra={"agents": []})

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("framework_observation"):
            raise ImportError("adapter unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert _deserialize_rich_run(observed) is None


def test_build_lanes_and_steps_happy_path_without_rich_framework_data():
    observed = _make_run(
        tool_calls=[
            _make_tool_call(
                tool_name="fetch_customer",
                agent_name="alice",
                arguments={"customer_id": 1},
                return_value={"summary": "customer loaded"},
            ),
            _make_tool_call(
                tool_name="fetch_case",
                agent_name="bob",
                arguments={"case_id": 9},
                return_value=[{"id": 9}],
            ),
        ],
        final_answer={"status": "complete"},
    )

    lanes, steps = _build_lanes_and_steps(observed)

    assert [lane.label for lane in lanes] == ["alice", "bob"]
    assert [step.kind for step in steps] == ["tool", "tool", "final"]
    assert steps[0].title == "1. fetch_customer"
    assert steps[1].title == "2. fetch_case"
    assert steps[-1].title == "Final Answer"
    assert steps[-1].lane_id == steps[1].lane_id
    assert steps[1].handoff_from_lane_id is None


def test_build_lanes_and_steps_adds_system_lane_and_final_when_no_tool_calls():
    observed = _make_run(tool_calls=[], final_answer="nothing to do", framework="demo")

    lanes, steps = _build_lanes_and_steps(observed)

    assert len(lanes) == 1
    assert lanes[0].lane_id == "system"
    assert lanes[0].label == "demo"
    assert len(steps) == 1
    assert steps[0].kind == "final"
    assert steps[0].lane_id == "system"


def test_build_lanes_and_steps_uses_rich_handoffs_final_output_and_errors(monkeypatch):
    class DummyFrameworkObservedRun:
        @classmethod
        def from_dict(cls, data):
            workflow = type(
                "Workflow",
                (),
                {
                    "handoffs": [
                        type(
                            "Handoff",
                            (),
                            {
                                "from_executor_id": "agent-1",
                                "to_executor_id": "agent-2",
                                "timestamp": "t1",
                                "payload_type": "review_packet",
                            },
                        )()
                    ]
                },
            )()
            return type(
                "RichObserved",
                (),
                {
                    "agents": [
                        type(
                            "Agent",
                            (),
                            {"agent_id": "agent-1", "agent_name": "Planner"},
                        )(),
                        type(
                            "Agent",
                            (),
                            {"agent_id": "agent-2", "agent_name": "Reviewer"},
                        )(),
                    ],
                    "workflow": workflow,
                    "final_output": {"summary": "rich final"},
                    "errors": [type("Err", (), {"message": "tool failed"})()],
                },
            )()

    import groundeval.diagram as diagram_module

    monkeypatch.setattr(
        diagram_module,
        "_deserialize_rich_run",
        lambda observed: diagram_module.RichRunView(
            agents=[
                diagram_module.RichAgent(agent_id="agent-1", agent_name="Planner"),
                diagram_module.RichAgent(agent_id="agent-2", agent_name="Reviewer"),
            ],
            handoffs=[
                diagram_module.RichHandoff(
                    from_executor_id="agent-1",
                    to_executor_id="agent-2",
                    timestamp="t1",
                    payload_type="review_packet",
                )
            ],
            final_output={"summary": "rich final"},
            errors=[type("Err", (), {"message": "tool failed"})()],
        ),
    )

    observed = _make_run(
        tool_calls=[
            _make_tool_call(tool_name="plan", agent_id="agent-1", agent_name="Planner"),
            _make_tool_call(
                tool_name="review", agent_id="agent-2", agent_name="Reviewer"
            ),
        ],
        final_answer={"summary": "plain final"},
        framework_extra={"any": "value"},
    )

    lanes, steps = _build_lanes_and_steps(observed)

    assert [lane.label for lane in lanes] == ["Planner", "Reviewer"]
    assert [step.kind for step in steps] == ["tool", "tool", "final", "error"]
    assert steps[1].handoff_from_lane_id == "agent-1"
    assert steps[1].handoff_label == "review_packet"
    assert steps[2].subtitle == "rich final"
    assert steps[3].title == "Error"
    assert steps[3].subtitle == "tool failed"


def test_render_observe_diagram_pdf_writes_non_empty_pdf(tmp_path):
    observed = _make_run(
        tool_calls=[
            _make_tool_call(
                tool_name="fetch_customer",
                agent_name="alice",
                arguments={"customer_id": 123, "include_history": True},
                return_value={"summary": "customer ready"},
            ),
            _make_tool_call(
                tool_name="search_docs",
                agent_name="alice",
                arguments={"query": "refund policy"},
                return_value=["doc-1", "doc-2"],
            ),
        ],
        final_answer={"message": "resolved"},
        run_id="pdf-run",
        total_latency_ms=456,
    )

    pdf_path = render_observe_diagram_pdf(tmp_path, observed)

    assert pdf_path == tmp_path / "observe_diagram.pdf"
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
    header = pdf_path.read_bytes()[:4]
    assert header == b"%PDF"


def test_render_observe_diagram_pdf_supports_multiple_pages(tmp_path):
    tool_calls = [
        _make_tool_call(
            tool_name=f"tool_{i}",
            agent_name="alice" if i % 2 == 0 else "bob",
            arguments={"index": i, "text": "x" * 30},
            return_value={"summary": f"result {i}"},
        )
        for i in range(1, 31)
    ]
    observed = _make_run(tool_calls=tool_calls, final_answer="done", run_id="paged-run")

    pdf_path = render_observe_diagram_pdf(tmp_path, observed)
    pdf_bytes = pdf_path.read_bytes()

    assert pdf_path.exists()
    assert pdf_bytes.startswith(b"%PDF")
    assert pdf_bytes.count(b"/Type /Page") >= 2


def test_write_observe_diagram_pdf_accepts_td_and_writes_file(tmp_path):
    observed = _make_run(
        tool_calls=[_make_tool_call(tool_name="fetch_customer")], final_answer="ok"
    )

    pdf_path = write_observe_diagram_pdf(tmp_path, observed, direction="TD")

    assert pdf_path.exists()
    assert pdf_path.name == "observe_diagram.pdf"
    assert pdf_path.read_bytes()[:4] == b"%PDF"


def test_write_observe_diagram_pdf_rejects_non_td_direction(tmp_path):
    observed = _make_run(tool_calls=[_make_tool_call()], final_answer="ok")

    with pytest.raises(
        ValueError, match="Only top-down direction 'TD' is currently supported."
    ):
        write_observe_diagram_pdf(tmp_path, observed, direction="LR")


def test_render_observe_diagram_pdf_overwrites_existing_file(tmp_path):
    observed = _make_run(
        tool_calls=[_make_tool_call(tool_name="fetch_customer")], final_answer="ok"
    )
    target = tmp_path / "observe_diagram.pdf"
    target.write_text("not a pdf", encoding="utf-8")

    pdf_path = render_observe_diagram_pdf(tmp_path, observed)

    assert pdf_path == target
    assert pdf_path.read_bytes()[:4] == b"%PDF"
    assert pdf_path.stat().st_size > len("not a pdf")
