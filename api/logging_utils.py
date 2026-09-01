"""
Structured (JSON) logging with a per-request correlation id.

Deliberately stdlib-only (logging.Formatter + logging.Filter +
contextvars) rather than adding structlog — "at minimum a consistent
JSON formatter" per the project's own improvement notes, and this is
plenty for a single-service API.

Wiring (see api/main.py):
  1. configure_logging() is called once at import time — replaces the
     root logger's handler with one that emits JSON.
  2. RequestIDMiddleware generates (or forwards) an X-Request-ID per
     request and stores it in a contextvar.
  3. RequestIdFilter (attached to the JSON handler) reads that
     contextvar into every LogRecord, so every log.warning()/exception()
     call anywhere in the app — including inside the POST /api/score
     background task, which runs in a copy of the same request's
     context — is automatically tagged with the request that caused it,
     with zero changes needed at each call site.
"""
import contextvars
import json
import logging
import uuid

request_id_var: "contextvars.ContextVar[str | None]" = contextvars.ContextVar("request_id", default=None)

# Attributes every stdlib LogRecord already has — used to detect which
# attributes on a record are "extra" fields a caller passed in
# (e.g. logger.warning(..., extra={"verdict_id": verdict_id})) and
# should be included in the JSON output verbatim.
_STANDARD_RECORD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
}


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key != "request_id":
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Configures the ROOT logger only — uvicorn's own access/error
    loggers (uvicorn.access, uvicorn.error) are independently configured
    with propagate=False, so this doesn't touch their console output.
    Every app-level logger (api.main, llm_agent, ...) propagates to
    root by default and picks up JSON formatting from here."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


class RequestIDMiddleware:
    """Pure ASGI middleware (not BaseHTTPMiddleware, which buffers the
    whole response and breaks streaming) — generates a request id per
    request, makes it available to every log call via request_id_var,
    and echoes it back as X-Request-ID so a client-reported error can be
    correlated to a specific log line."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(b"x-request-id")
        request_id = incoming.decode() if incoming else str(uuid.uuid4())
        token = request_id_var.set(request_id)

        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                message["headers"].append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            request_id_var.reset(token)
