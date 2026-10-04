"""Overlap model I/O without moving simulation callbacks off their owner thread.

Callbacks are cooperative greenlets. Only decorated, blocking model requests
enter the thread pool; tools, SUMO, per-agent state and rendering stay on the
simulation thread. Every batch is a barrier before the next physics step.
"""

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from functools import wraps

from greenlet import greenlet, getcurrent


@dataclass
class _OwnerCall:
    function: object


def model_io(function):
    """Mark the complete request/retry loop as independent, blocking model I/O."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        current = getcurrent()
        scheduler = getattr(current, "model_scheduler", None)
        if scheduler is None:
            return function(*args, **kwargs)
        future = scheduler.executor.submit(function, *args, **kwargs)
        current.parent.switch(future)
        return future.result()
    return wrapped


def on_simulation_owner(function, *args, **kwargs):
    """Keep thread/greenlet-affine clients (notably Playwright) on the owner."""
    current = getcurrent()
    if getattr(current, "model_scheduler", None) is None:
        return function(*args, **kwargs)
    return current.parent.switch(_OwnerCall(lambda: function(*args, **kwargs)))


class ModelCallScheduler:
    def __init__(self, max_workers=4):
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("max_parallel_model_calls must be a positive integer")
        self.max_workers = max_workers
        self.executor = None

    def __enter__(self):
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="vehicle-model-io")
        return self

    def __exit__(self, *exc):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def run(self, jobs):
        """Run a frozen batch; drain siblings even when one callback fails."""
        jobs = list(jobs)
        if self.max_workers == 1 or len(jobs) < 2:
            return [job() for job in jobs]
        pending = {}
        results = [None] * len(jobs)
        errors = {}

        def advance(index, task):
            try:
                yielded = task.switch()
                while not task.dead and isinstance(yielded, _OwnerCall):
                    try:
                        result = yielded.function()
                    except Exception as exc:
                        yielded = task.throw(exc)
                    else:
                        yielded = task.switch(result)
                if task.dead:
                    results[index] = yielded
                else:
                    pending[index] = (task, yielded)
            except Exception as exc:
                errors[index] = exc

        for index, job in enumerate(jobs):
            task = greenlet(job)
            task.model_scheduler = self
            advance(index, task)
        while pending:
            wait([future for _, future in pending.values()], return_when=FIRST_COMPLETED)
            for index in list(pending):
                task, future = pending[index]
                if future.done():
                    del pending[index]
                    advance(index, task)
        if errors:
            raise errors[min(errors)]
        return results
