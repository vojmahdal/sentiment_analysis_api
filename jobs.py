"""
In-memory tracker for background /ingest jobs.

A large /ingest batch can take a while (see pipeline.py and the README
section on batched inference), so it is run in a background thread instead
of blocking the request. This module tracks each job's progress so the
dashboard can poll it and show a progress bar and elapsed-time timer.

Job state lives only in process memory - it is intentionally simple (no
persistence, no cross-process sharing), consistent with the rest of this
prototype. A restart of the server loses in-flight job status, which the
frontend treats as an error (job id no longer known).
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# Old finished jobs are never explicitly deleted (kept simple), but are
# capped so a long-running server doesn't accumulate unbounded memory.
_MAX_JOBS = 200


def create_job(total: int) -> str:
    """Register a new job and return its id."""
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",  # running | done | error
            "processed": 0,
            "total": total,
            "started_at": time.time(),
            "finished_at": None,
            "result": None,
            "error": None,
        }
        while len(_jobs) > _MAX_JOBS:
            _jobs.pop(next(iter(_jobs)))
    return job_id


def progress_callback(job_id: str) -> Callable[[int], None]:
    """Return a callback suitable for ``pipeline.process_batch(on_progress=...)``."""

    def _update(processed: int) -> None:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["processed"] = processed

    return _update


def finish_job(job_id: str, result: dict[str, Any]) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["status"] = "done"
            job["result"] = result
            job["finished_at"] = time.time()


def fail_job(job_id: str, error: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["status"] = "error"
            job["error"] = error
            job["finished_at"] = time.time()


def get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None
