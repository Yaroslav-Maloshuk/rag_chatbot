# Project Structure

This repository now follows a clearer layered layout while preserving backward compatibility.

## Top-Level Layout

```text
rag_chatbot/
  app/
    bootstrap/          # Startup wiring (FastAPI app factory, worker bootstrap)
    presentation/       # Delivery layer (HTTP routes, request dependencies)
    core/               # Configuration, logging, observability
    db/                 # DB engine/session/models initialization
    repositories/       # Data access abstractions
    services/           # Application services / orchestration
    tasks/              # Async jobs (Celery tasks)
    schemas/            # API/data schemas
    utils/              # Shared utility functions
    api/                # Backward-compatible import shims (legacy path)
  frontend/             # React + Vite UI
  tests/                # Backend tests
  scripts/              # Developer/operator scripts
  docker/               # Infra bootstrap SQL and related assets
  docs/                 # Engineering docs
```

## Layer Responsibilities

- `bootstrap`: application assembly and entrypoint composition.
- `presentation`: API endpoint contracts and external interface concerns.
- `services`: business orchestration and workflow coordination.
- `repositories` + `db`: persistence access and storage integration.
- `tasks`: asynchronous/background execution.
- `core`: cross-cutting infrastructure (settings, logging, metrics).

## Backward Compatibility

Legacy `app.api.*` imports are intentionally preserved via thin wrappers that delegate to:

- `app.presentation.http.deps`
- `app.presentation.http.router`
- `app.presentation.http.routes.*`

This allows gradual migration without breaking runtime scripts and existing imports.

