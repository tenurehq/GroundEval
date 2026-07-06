# Understanding `diagram.py`

`diagram.py` is the part of GroundEval that turns an observed agent run into a PDF diagram. Its job is to take the structured observation data from a run and turn it into a visual timeline that's much easier to review than raw JSON.

## What it's used for

The module generates `observe_diagram.pdf`, a report that shows:

- which agents were involved in a run
- which tool calls happened and in what order
- when control appeared to move from one agent to another
- what the final answer was
- whether any errors showed up along the way

This makes it useful for debugging, reviewing agent behavior, and sharing a readable summary of a multi-step or multi-agent run with someone else.

## The main inputs

The diagram is built from an observed run: the run id, framework name, agent class, the tool calls that happened, the final answer, total latency, and (optionally) some richer framework-specific data if it's available.

In other words, `diagram.py` doesn't observe agents itself — it just takes observation data that's already been collected and turns it into a PDF.

## The main output

The output is a file called `observe_diagram.pdf`, saved to a target directory. By default, it renders top-down.

## How the diagram is structured

The PDF is laid out like a lightweight workflow view.

### 1. Header
The top of the document includes summary info like the run id, framework, agent class, number of tool calls, latency, number of agents observed, and the artifact name — enough context before diving into the step-by-step flow.

### 2. Legend
A small legend explains what the visuals mean: tool call, final answer, handoff, and error.

### 3. Agent lanes
Activity is grouped into vertical lanes, one per agent (or a general system lane if no agent info is available). When richer data is available, it's used to give agents clearer, more accurate labels.

### 4. Steps
Each step in the run shows up as a rounded box inside a lane. A step can be a tool call, the final answer, or an error.

Tool call boxes include a title (like "1. fetch_customer"), a short summary of what was passed in, and a short summary of what came back.

The final answer gets its own box after the tool calls — using the richer version of the answer when one is available. If any errors were recorded, up to three of them are shown after that.

## How it handles multi-agent runs

One of the more helpful parts of the module is that it tries to show agent-to-agent movement. When consecutive tool calls happen in different lanes, the diagram checks whether there was an actual handoff between agents. If so, it draws a connector between the lanes and labels it, so you can see collaboration flow across agents — not just a flat list of calls.

## Rich framework support

When richer, framework-specific observation data is available, `diagram.py` will use it to fill in more detail — clearer agent identities, explicit handoffs, a richer final answer, and any recorded errors.

If that richer data isn't available or can't be read for some reason, the diagram still renders using the basic observed run data. So the PDF is always generated, just with more detail when it can be.

## When this file is used in the project

This module runs as part of generating observation output. When the CLI writes out observation artifacts, one of them is this diagram PDF.

Practically speaking, `diagram.py` is part of the reporting layer of GroundEval. It doesn't score the run and it doesn't collect the events — it just presents what was observed in a visual format that's easy for a human to read.

## In plain English

If capturing what happened is one job, making that run understandable at a glance is this module's job.

It turns agent activity into a PDF that answers questions like:

- What tools were called?
- In what order?
- Which agent made each call?
- Did work move between agents?
- What final answer came out?
- Were any errors recorded?

That makes it especially useful for debugging, demos, audits, and reviewing complex multi-agent workflows without having to read raw JSON files.