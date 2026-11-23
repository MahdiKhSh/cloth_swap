"""Background jobs on a single GPU worker thread, with progress, logs and cancelling."""
from __future__ import annotations

import collections
import logging
import queue
import threading
import time
import traceback
import uuid
from contextlib import contextmanager

log = logging.getLogger(__name__)


class Cancelled(Exception):
    pass


class GpuBusy(Exception):
    pass


class Job:
    def __init__(self, project_id: str, kind: str, title: str, fn, gpu: bool = True):
        self.id = uuid.uuid4().hex[:10]
        self.project_id = project_id
        self.kind = kind
        self.title = title
        self.fn = fn
        self.gpu = gpu
        self.status = "queued"
        self.progress = 0.0
        self.message = ""
        self.error = None
        self.result = None
        self.created = time.time()
        self.started = None
        self.finished = None
        self.logs = collections.deque(maxlen=200)
        self.cancel_requested = False
        self.process = None  # a subprocess to terminate on cancel

    def set(self, progress: float | None = None, message: str | None = None) -> None:
        if progress is not None:
            self.progress = max(0.0, min(1.0, float(progress)))
        if message is not None and message != self.message:
            self.message = message
            self.log(message)
        self.check()

    def log(self, line: str) -> None:
        self.logs.append(f"{time.strftime('%H:%M:%S')} {line}")

    def check(self) -> None:
        if self.cancel_requested:
            raise Cancelled()

    def to_dict(self, logs: bool = False) -> dict:
        d = {k: getattr(self, k) for k in ("id", "project_id", "kind", "title", "status", "progress",
                                           "message", "error", "result", "created", "started", "finished")}
        if logs:
            d["logs"] = list(self.logs)
        return d


class JobManager:
    def __init__(self, before_job=None):
        self.jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self.queue: queue.Queue[Job] = queue.Queue()
        self.gpu = threading.Lock()
        self.before_job = before_job
        threading.Thread(target=self._worker, daemon=True, name="gpu-worker").start()

    def submit(self, project_id: str, kind: str, title: str, fn, gpu: bool = True) -> Job:
        """Queue fn(job). GPU jobs run one at a time on the worker; others get their own thread."""
        with self._lock:
            # Import, tracking, mask edits and generation share on-disk files.
            # Different job kinds must never run concurrently on one project.
            if any(j.project_id == project_id and j.status in ("queued", "running")
                   for j in self.jobs.values()):
                raise ValueError("Another step is already running for this project")
            job = Job(project_id, kind, title, fn, gpu=gpu)
            self.jobs[job.id] = job
        if gpu:
            self.queue.put(job)
        else:
            threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self.jobs[job_id]
            job.cancel_requested = True
            if job.status == "queued":
                job.status = "cancelled"
                job.finished = time.time()
        if job.process is not None and job.process.poll() is None:
            import psutil
            try:
                parent = psutil.Process(job.process.pid)
                children = parent.children(recursive=True)
                for child in reversed(children):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                parent.kill()
            except (psutil.Error, OSError):
                pass
        return job

    def for_project(self, project_id: str) -> list[Job]:
        with self._lock:
            return sorted((j for j in self.jobs.values() if j.project_id == project_id), key=lambda j: j.created)

    def busy(self, gpu_only: bool = False) -> Job | None:
        with self._lock:
            return next((j for j in self.jobs.values()
                         if j.status == "running" and (not gpu_only or j.gpu)), None)

    def active_for_project(self, project_id: str) -> bool:
        with self._lock:
            return any(j.project_id == project_id and j.status in ("queued", "running")
                       for j in self.jobs.values())

    @contextmanager
    def interactive(self):
        """For quick GPU calls made inside a request; refuses while a job owns the GPU, but waits
        for other short holders (model warm-up, another click)."""
        if self.busy(gpu_only=True) is not None or not self.gpu.acquire(timeout=180):
            raise GpuBusy()
        try:
            yield
        finally:
            self.gpu.release()

    def _worker(self):
        while True:
            job = self.queue.get()
            try:
                if job.status != "cancelled":
                    with self.gpu:
                        if job.status != "cancelled":
                            self._run(job)
            finally:
                self.queue.task_done()

    def _run(self, job: Job) -> None:
        if job.cancel_requested:
            job.status = "cancelled"
            job.finished = time.time()
            return
        job.status = "running"
        job.started = time.time()
        try:
            job.check()
            if self.before_job:
                self.before_job(job)
            job.result = job.fn(job)
            job.check()
            job.status = "done"
            job.progress = 1.0
        except Cancelled:
            job.status = "cancelled"
            job.log("cancelled")
        except Exception as exc:  # reported to the UI
            if job.cancel_requested:
                job.status = "cancelled"
            else:
                job.status = "failed"
                job.error = str(exc)
                job.log(traceback.format_exc())
                log.exception("Job %s (%s) failed", job.id, job.kind)
        finally:
            job.finished = time.time()
            job.process = None
