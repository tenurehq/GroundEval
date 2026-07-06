from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reportlab.lib.colors import HexColor, black
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from .observe import ObservedRun, ObservedToolCall


@dataclass
class AgentLane:
    lane_id: str
    label: str
    order: int


@dataclass
class DiagramStep:
    step_index: int
    lane_id: str
    title: str
    subtitle: str = ""
    detail: str = ""
    kind: str = "tool"
    handoff_from_lane_id: str | None = None
    handoff_label: str = ""


@dataclass
class RichAgent:
    agent_id: str
    agent_name: str | None = None


@dataclass
class RichHandoff:
    from_executor_id: str
    to_executor_id: str
    timestamp: str | None = None
    payload_type: str | None = None


@dataclass
class RichRunView:
    agents: list[RichAgent]
    handoffs: list[RichHandoff]
    final_output: Any
    errors: list[Any]


PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN_X = 36
MARGIN_TOP = 36
MARGIN_BOTTOM = 36
HEADER_H = 74
LEGEND_H = 56
LANE_HEADER_H = 22
STEP_H = 74
STEP_GAP = 18
LANE_GAP = 18
LANE_MIN_W = 190
FONT = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
TOOL_FILL = HexColor("#E8F1FF")
TOOL_STROKE = HexColor("#2F6FEB")
FINAL_FILL = HexColor("#FFF6E5")
FINAL_STROKE = HexColor("#C77700")
ERROR_FILL = HexColor("#FDECEC")
ERROR_STROKE = HexColor("#C62828")
HANDOFF_COLOR = HexColor("#7E57C2")
LANE_FILL = HexColor("#F3F4F6")
LANE_STROKE = HexColor("#D1D5DB")
LEGEND_FILL = HexColor("#FAFAFA")
LEGEND_STROKE = HexColor("#D9D9D9")
CONNECTOR_COLOR = HexColor("#94A3B8")
TEXT_MUTED = HexColor("#6B7280")


def _safe_json(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


def _truncate(text: str, limit: int) -> str:
    text = (text or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _summarize_arguments(arguments: dict[str, Any] | None, limit: int = 56) -> str:
    if not arguments:
        return ""
    parts = []
    for key, value in list(arguments.items())[:3]:
        rendered = _truncate(_safe_json(value), 20)
        parts.append(f"{key}={rendered}")
    return _truncate(", ".join(parts), limit)


def _summarize_return_value(value: Any, limit: int = 64) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("summary", "title", "name", "text", "message", "content"):
            if key in value and value.get(key):
                return _truncate(str(value[key]), limit)
    if isinstance(value, list):
        if not value:
            return "empty result"
        if len(value) == 1:
            return _truncate(f"1 result: {_safe_json(value[0])}", limit)
        return f"{len(value)} results"
    return _truncate(_safe_json(value), limit)


def _wrap_text(
    text: str, max_width: float, font_name: str, font_size: int
) -> list[str]:
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _lane_label(agent_id: str | None, agent_name: str | None) -> str:
    if agent_name and agent_id and agent_name != agent_id:
        return agent_name
    if agent_name:
        return agent_name
    if agent_id:
        return agent_id
    return "Agent"


def _deserialize_rich_run(observed: ObservedRun) -> RichRunView | None:
    if not observed.framework_extra:
        return None
    try:
        from .framework_adapters.framework_observation import (
            ObservedRun as FrameworkObservedRun,
        )

        rich = FrameworkObservedRun.from_dict(observed.framework_extra)
        agents = [
            RichAgent(agent_id=a.agent_id, agent_name=a.agent_name)
            for a in (rich.agents or [])
        ]
        handoffs = [
            RichHandoff(
                from_executor_id=h.from_executor_id,
                to_executor_id=h.to_executor_id,
                timestamp=h.timestamp,
                payload_type=h.payload_type,
            )
            for h in ((rich.workflow.handoffs if rich.workflow else []) or [])
        ]
        return RichRunView(
            agents=agents,
            handoffs=handoffs,
            final_output=rich.final_output,
            errors=list(rich.errors or []),
        )
    except Exception:
        return None


def _resolve_lane_id(tc: ObservedToolCall, known_agents: dict[str, str]) -> str:
    if tc.agent_id:
        return tc.agent_id
    if tc.agent_name:
        for aid, name in known_agents.items():
            if name == tc.agent_name:
                return aid
        return tc.agent_name
    return "system"


def _build_lanes_and_steps(
    observed: ObservedRun,
) -> tuple[list[AgentLane], list[DiagramStep]]:
    rich = _deserialize_rich_run(observed)
    known_agents: dict[str, str] = {}
    lane_ids_in_order: list[str] = []
    if rich:
        for agent in rich.agents:
            aid = agent.agent_id or agent.agent_name or "system"
            known_agents[aid] = agent.agent_name or aid
    for tc in observed.tool_calls:
        lane_id = _resolve_lane_id(tc, known_agents)
        if lane_id not in lane_ids_in_order:
            lane_ids_in_order.append(lane_id)
            known_agents.setdefault(lane_id, tc.agent_name or tc.agent_id or "Agent")
    if not lane_ids_in_order:
        lane_ids_in_order.append("system")
        known_agents.setdefault("system", observed.framework)
    lanes = [
        AgentLane(
            lane_id=lane_id,
            label=_lane_label(lane_id, known_agents.get(lane_id)),
            order=i,
        )
        for i, lane_id in enumerate(lane_ids_in_order)
    ]
    handoff_by_pair: dict[tuple[str, str], RichHandoff] = {}
    if rich:
        for handoff in rich.handoffs:
            handoff_by_pair[(handoff.from_executor_id, handoff.to_executor_id)] = (
                handoff
            )
    steps: list[DiagramStep] = []
    prev_lane_id: str | None = None
    for i, tc in enumerate(observed.tool_calls, start=1):
        lane_id = _resolve_lane_id(tc, known_agents)
        handoff_from_lane_id = None
        handoff_label = ""
        if prev_lane_id and prev_lane_id != lane_id:
            handoff = handoff_by_pair.get((prev_lane_id, lane_id))
            if handoff is not None:
                handoff_from_lane_id = prev_lane_id
                handoff_label = str(handoff.payload_type or "handoff")
        steps.append(
            DiagramStep(
                step_index=i,
                lane_id=lane_id,
                title=f"{i}. {tc.tool_name}",
                subtitle=_summarize_arguments(tc.arguments),
                detail=_summarize_return_value(tc.return_value),
                kind="tool",
                handoff_from_lane_id=handoff_from_lane_id,
                handoff_label=handoff_label,
            )
        )
        prev_lane_id = lane_id
    final_output = observed.final_answer
    if rich and rich.final_output is not None:
        final_output = rich.final_output
    final_lane_id = prev_lane_id or lanes[0].lane_id
    steps.append(
        DiagramStep(
            step_index=len(steps) + 1,
            lane_id=final_lane_id,
            title="Final Answer",
            subtitle=_summarize_return_value(final_output, limit=84),
            kind="final",
        )
    )
    if rich and rich.errors:
        for error in rich.errors[:3]:
            steps.append(
                DiagramStep(
                    step_index=len(steps) + 1,
                    lane_id=final_lane_id,
                    title="Error",
                    subtitle=_truncate(getattr(error, "message", ""), 72),
                    kind="error",
                )
            )
    return lanes, steps


def _draw_header(
    pdf: canvas.Canvas,
    observed: ObservedRun,
    lanes: list[AgentLane],
    steps: list[DiagramStep],
) -> None:
    pdf.setFillColor(black)
    pdf.setFont(FONT_BOLD, 16)
    pdf.drawString(MARGIN_X, PAGE_HEIGHT - MARGIN_TOP, "GroundEval Observed Agent Run")
    pdf.setFont(FONT, 9)
    meta_y = PAGE_HEIGHT - MARGIN_TOP - 16
    pdf.drawString(MARGIN_X, meta_y, f"Run ID: {observed.run_id}")
    pdf.drawString(MARGIN_X, meta_y - 12, f"Framework: {observed.framework}")
    pdf.drawString(MARGIN_X, meta_y - 24, f"Agent Class: {observed.agent_class}")
    pdf.drawString(
        MARGIN_X,
        meta_y - 36,
        f"Tool Calls: {max(0, len([s for s in steps if s.kind == 'tool']))}",
    )
    pdf.drawString(
        MARGIN_X + 260, meta_y, f"Latency: {observed.total_latency_ms:.0f}ms"
    )
    pdf.drawString(MARGIN_X + 260, meta_y - 12, f"Agents Observed: {len(lanes)}")
    pdf.drawString(MARGIN_X + 260, meta_y - 24, "Mode: Observe")
    pdf.drawString(MARGIN_X + 260, meta_y - 36, "Artifact: observe_diagram.pdf")


def _draw_legend(pdf: canvas.Canvas, top_y: float) -> None:
    legend_x = MARGIN_X
    legend_y = top_y - LEGEND_H
    legend_w = PAGE_WIDTH - (2 * MARGIN_X)
    pdf.setStrokeColor(LEGEND_STROKE)
    pdf.setFillColor(LEGEND_FILL)
    pdf.roundRect(legend_x, legend_y, legend_w, LEGEND_H, 6, stroke=1, fill=1)
    items = [
        ("Tool call", TOOL_FILL, TOOL_STROKE),
        ("Final answer", FINAL_FILL, FINAL_STROKE),
        ("Handoff", HexColor("#F4ECFF"), HANDOFF_COLOR),
        ("Error", ERROR_FILL, ERROR_STROKE),
    ]
    x = legend_x + 12
    y = legend_y + 22
    pdf.setFont(FONT, 8)
    for label, fill, stroke in items:
        pdf.setFillColor(fill)
        pdf.setStrokeColor(stroke)
        pdf.roundRect(x, y, 14, 10, 2, stroke=1, fill=1)
        pdf.setFillColor(black)
        pdf.drawString(x + 20, y + 2, label)
        x += 104


def _lane_geometry(
    lanes: list[AgentLane],
) -> dict[str, tuple[float, float, float, float]]:
    usable_w = PAGE_WIDTH - (2 * MARGIN_X)
    lane_count = max(1, len(lanes))
    lane_w = max(LANE_MIN_W, (usable_w - ((lane_count - 1) * LANE_GAP)) / lane_count)
    total_w = lane_count * lane_w + (lane_count - 1) * LANE_GAP
    start_x = MARGIN_X + max(0, (usable_w - total_w) / 2)
    geometry = {}
    for i, lane in enumerate(lanes):
        x = start_x + i * (lane_w + LANE_GAP)
        geometry[lane.lane_id] = (x, 0, lane_w, 0)
    return geometry


def _draw_lane_headers(
    pdf: canvas.Canvas,
    lanes: list[AgentLane],
    lane_boxes: dict[str, tuple[float, float, float, float]],
    top_y: float,
) -> float:
    y = top_y - LANE_HEADER_H
    for lane in lanes:
        x, _, w, _ = lane_boxes[lane.lane_id]
        pdf.setFillColor(LANE_FILL)
        pdf.setStrokeColor(LANE_STROKE)
        pdf.roundRect(x, y, w, LANE_HEADER_H, 4, stroke=1, fill=1)
        pdf.setFillColor(black)
        pdf.setFont(FONT_BOLD, 9)
        pdf.drawString(x + 8, y + 7, _truncate(lane.label, 28))
    return y - 10


def _step_height(step: DiagramStep) -> float:
    lines = 3
    if step.kind in {"final", "error"}:
        lines = 2
    return max(STEP_H, 18 + (lines * 12))


def _paginate_steps(
    steps: list[DiagramStep], available_h: float
) -> list[list[DiagramStep]]:
    pages: list[list[DiagramStep]] = []
    current: list[DiagramStep] = []
    used = 0.0
    for step in steps:
        h = _step_height(step) + STEP_GAP
        if current and used + h > available_h:
            pages.append(current)
            current = [step]
            used = h
        else:
            current.append(step)
            used += h
    if current:
        pages.append(current)
    return pages


def _draw_vertical_connector(
    pdf: canvas.Canvas, x: float, y1: float, y2: float, color: Any
) -> None:
    pdf.setStrokeColor(color)
    pdf.setLineWidth(1.5)
    pdf.line(x, y1, x, y2)
    pdf.line(x, y2, x - 4, y2 + 6)
    pdf.line(x, y2, x + 4, y2 + 6)


def _draw_handoff(
    pdf: canvas.Canvas,
    from_box: tuple[float, float, float, float],
    to_box: tuple[float, float, float, float],
    label: str,
) -> None:
    fx, fy, fw, fh = from_box
    tx, ty, tw, th = to_box
    start_x = fx + fw / 2
    start_y = fy
    end_x = tx + tw / 2
    end_y = ty + th
    mid_y = (start_y + end_y) / 2
    pdf.setStrokeColor(HANDOFF_COLOR)
    pdf.setLineWidth(1.3)
    pdf.line(start_x, start_y, start_x, mid_y)
    pdf.line(start_x, mid_y, end_x, mid_y)
    pdf.line(end_x, mid_y, end_x, end_y)
    pdf.line(end_x, end_y, end_x - 4, end_y + 6)
    pdf.line(end_x, end_y, end_x + 4, end_y + 6)
    if label:
        pdf.setFillColor(HANDOFF_COLOR)
        pdf.setFont(FONT, 7)
        pdf.drawCentredString((start_x + end_x) / 2, mid_y + 4, _truncate(label, 18))


def _draw_step(
    pdf: canvas.Canvas, step: DiagramStep, x: float, y: float, w: float, h: float
) -> tuple[float, float, float, float]:
    if step.kind == "tool":
        fill = TOOL_FILL
        stroke = TOOL_STROKE
    elif step.kind == "final":
        fill = FINAL_FILL
        stroke = FINAL_STROKE
    elif step.kind == "error":
        fill = ERROR_FILL
        stroke = ERROR_STROKE
    else:
        fill = HexColor("#F5F5F5")
        stroke = HexColor("#757575")
    pdf.setFillColor(fill)
    pdf.setStrokeColor(stroke)
    pdf.setLineWidth(1.4)
    pdf.roundRect(x, y, w, h, 8, stroke=1, fill=1)
    text_x = x + 8
    text_y = y + h - 14
    pdf.setFillColor(black if step.kind != "error" else HexColor("#8B0000"))
    pdf.setFont(FONT_BOLD, 10)
    for line in _wrap_text(step.title, w - 16, FONT_BOLD, 10)[:2]:
        pdf.drawString(text_x, text_y, line)
        text_y -= 12
    pdf.setFont(FONT, 8)
    for line in _wrap_text(step.subtitle, w - 16, FONT, 8)[:2]:
        pdf.drawString(text_x, text_y, line)
        text_y -= 10
    if step.detail and step.kind == "tool":
        for line in _wrap_text(step.detail, w - 16, FONT, 8)[:1]:
            pdf.drawString(text_x, text_y, line)
            text_y -= 10
    return (x, y, w, h)


def render_observe_diagram_pdf(output_dir: str | Path, observed: ObservedRun) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "observe_diagram.pdf"
    lanes, steps = _build_lanes_and_steps(observed)
    lane_boxes = _lane_geometry(lanes)
    available_h = (
        PAGE_HEIGHT
        - MARGIN_TOP
        - HEADER_H
        - LEGEND_H
        - LANE_HEADER_H
        - MARGIN_BOTTOM
        - 24
    )
    paged_steps = _paginate_steps(steps, available_h)
    pdf = canvas.Canvas(str(pdf_path), pagesize=letter)
    pdf.setTitle(f"GroundEval Observe Diagram {observed.run_id}")
    for page_index, page_steps in enumerate(paged_steps, start=1):
        _draw_header(pdf, observed, lanes, steps)
        legend_top = PAGE_HEIGHT - MARGIN_TOP - HEADER_H + 6
        _draw_legend(pdf, legend_top)
        lane_top = PAGE_HEIGHT - MARGIN_TOP - HEADER_H - LEGEND_H - 8
        content_y = _draw_lane_headers(pdf, lanes, lane_boxes, lane_top)
        last_box_by_lane: dict[str, tuple[float, float, float, float]] = {}
        row_top_y = content_y
        for step in page_steps:
            x, _, w, _ = lane_boxes[step.lane_id]
            h = _step_height(step)
            box_y = row_top_y - h
            prev_box_same_lane = last_box_by_lane.get(step.lane_id)
            if prev_box_same_lane:
                center_x = x + (w / 2)
                _draw_vertical_connector(
                    pdf, center_x, prev_box_same_lane[1], box_y + h, CONNECTOR_COLOR
                )
            box = _draw_step(pdf, step, x, box_y, w, h)
            if step.handoff_from_lane_id:
                from_box = last_box_by_lane.get(step.handoff_from_lane_id)
                if from_box:
                    _draw_handoff(pdf, from_box, box, step.handoff_label)
            last_box_by_lane[step.lane_id] = box
            row_top_y = box_y - STEP_GAP
        pdf.setFillColor(TEXT_MUTED)
        pdf.setFont(FONT, 8)
        pdf.drawRightString(
            PAGE_WIDTH - MARGIN_X, 18, f"Page {page_index} of {len(paged_steps)}"
        )
        pdf.showPage()
    pdf.save()
    return pdf_path


def write_observe_diagram_pdf(
    output_dir: str | Path, observed: ObservedRun, direction: str = "TD"
) -> Path:
    if (direction or "TD").upper() != "TD":
        raise ValueError("Only top-down direction 'TD' is currently supported.")
    return render_observe_diagram_pdf(output_dir=output_dir, observed=observed)
