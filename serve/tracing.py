"""End-to-end tracing for the search pipeline.

Wrap any pipeline function in @traced and it logs, on every call:
  - entry: the arguments it received
  - exit:  the value it returned, plus wall-clock duration
  - errors: the exception type and message, with duration, before re-raising

Two problems this has to solve to be useful rather than noise:

1. Values here are huge. The pipeline passes around a (129971, 384) embeddings
   matrix, 100-element candidate id lists, and paragraph-length descriptions.
   Logging them verbatim buries the signal, so `summarize()` renders shapes and
   counts instead of contents.

2. Stages need correlating. One search fans out across five functions, and
   concurrent requests interleave in the log. A trace id in a ContextVar ties
   them together without threading an argument through every signature --
   which also means it works identically under FastAPI and Gradio, and stays
   correct across asyncio tasks and threads.

Enable with WINE_TRACE=1 (or call configure_tracing()).
"""

import contextvars
import functools
import logging
import os
import time
import uuid
from typing import Any, Callable, TypeVar

logger = logging.getLogger("wine.trace")

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
_depth: contextvars.ContextVar[int] = contextvars.ContextVar("depth", default=0)

MAX_STR = 120
MAX_ITEMS = 6

F = TypeVar("F", bound=Callable[..., Any])


def configure_tracing(level: int = logging.INFO) -> None:
    """Attach a handler to the trace logger. Idempotent."""
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def new_trace(prefix: str = "") -> str:
    """Start a new trace scope. Call once per incoming request."""
    trace_id = f"{prefix}{uuid.uuid4().hex[:8]}"
    _trace_id.set(trace_id)
    _depth.set(0)
    return trace_id


def summarize(value: Any, _nested: bool = False) -> str:
    """Render a value compactly: shapes and counts, never full contents."""
    # numpy is imported lazily so tracing stays usable in contexts without it
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        if isinstance(value, np.generic):
            return repr(value.item())
    except ImportError:
        pass

    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)

    if isinstance(value, str):
        if len(value) <= MAX_STR:
            return repr(value)
        return f"{value[:MAX_STR]!r}... ({len(value)} chars)"

    if isinstance(value, dict):
        # Filter dicts are mostly Nones; showing only what's set is the useful view.
        set_items = {k: v for k, v in value.items() if v is not None}
        omitted = len(value) - len(set_items)
        body = ", ".join(f"{k}={summarize(v, True)}" for k, v in list(set_items.items())[:MAX_ITEMS])
        if len(set_items) > MAX_ITEMS:
            body += f", +{len(set_items) - MAX_ITEMS} more"
        suffix = f" (+{omitted} unset)" if omitted else ""
        return "{" + body + "}" + suffix

    if isinstance(value, (list, tuple, set)):
        kind = type(value).__name__
        if not value:
            return f"{kind}(empty)"
        head = ", ".join(summarize(v, True) for v in list(value)[:MAX_ITEMS])
        more = f", +{len(value) - MAX_ITEMS} more" if len(value) > MAX_ITEMS else ""
        return f"{kind}[{head}{more}] (n={len(value)})"

    return f"{type(value).__name__}(...)"


def _log(arrow: str, message: str) -> None:
    indent = "  " * _depth.get()
    logger.info("[%s] %s%s %s", _trace_id.get(), indent, arrow, message)


def traced(func: F | None = None, *, root: bool = False) -> Any:
    """Log inputs on entry and the return value plus duration on exit.

    Usable bare (`@traced`) or with options (`@traced(root=True)`). Set root on
    a request entry point: it allocates the trace id *before* the entry line is
    logged, so the whole request shares one id. Calling new_trace() inside the
    function body instead would be too late -- entry is logged first, and would
    carry the previous request's id.
    """

    def decorator(target: F) -> F:
        @functools.wraps(target)
        def wrapper(*args, **kwargs):
            if not logger.isEnabledFor(logging.INFO):
                return target(*args, **kwargs)

            if root:
                new_trace()

            parts = [summarize(a) for a in args]
            parts += [f"{k}={summarize(v)}" for k, v in kwargs.items()]
            _log("→", f"{target.__name__}({', '.join(parts)})")

            _depth.set(_depth.get() + 1)
            started = time.perf_counter()
            try:
                result = target(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                _depth.set(_depth.get() - 1)
                _log("✗", f"{target.__name__} raised {type(exc).__name__}: {exc} [{elapsed_ms:.0f}ms]")
                raise
            elapsed_ms = (time.perf_counter() - started) * 1000
            _depth.set(_depth.get() - 1)
            _log("←", f"{target.__name__} = {summarize(result)} [{elapsed_ms:.0f}ms]")
            return result

        return wrapper  # type: ignore[return-value]

    return decorator(func) if func is not None else decorator


if os.environ.get("WINE_TRACE"):
    configure_tracing()
