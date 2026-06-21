FROM python:3.13-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv sync --group dev

ENTRYPOINT ["uv", "run", "orgforge"]