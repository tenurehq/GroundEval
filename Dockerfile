FROM python:3.13-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config

ARG AGENT_FRAMEWORK=""

ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_GROUNDEVAL=0.0.0

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
      --group dev${AGENT_FRAMEWORK:+ --group ${AGENT_FRAMEWORK}}

ENTRYPOINT ["uv", "run", "groundeval"]