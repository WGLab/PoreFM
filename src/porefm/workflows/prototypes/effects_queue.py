"""Bounded FIFO handoff to one CPU decoder/writer thread."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor


class BoundedScoreQueue:
    """Overlap a producer with one consumer while bounding outstanding work.

    ``consume(payload)`` runs serially on one thread and should not return a
    large result. ``max_items`` and ``max_bytes`` include the currently running
    item. Byte accounting is supplied by the producer and should cover numeric
    payload and metadata; Python allocator overhead is additional. An oversized
    single item raises rather than waiting forever. No tasks are reordered.

    Consumer exceptions are propagated to the producer on submission, drain,
    or normal context exit. Exceptional exit cancels pending work, waits for
    the running item to finish, and joins the thread. The producer must release
    its own iterator/resources in a finally block. No CUDA tensors belong in
    submitted payloads.
    """

    def __init__(self, consume, *, max_items=8, max_bytes=256 * 1024 ** 2):
        if max_items < 1 or max_bytes < 1:
            raise ValueError('score queue item and byte limits must be positive')
        self.consume = consume
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.pending_bytes = 0
        self.peak_items = 0
        self.peak_bytes = 0
        self._pending = deque()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='effect-decode')

    def __enter__(self):
        return self

    def _finish_oldest(self):
        future, size = self._pending.popleft()
        self.pending_bytes -= size
        future.result()

    def check(self):
        """Release completed FIFO items and surface any consumer exception."""
        while self._pending and self._pending[0][0].done():
            self._finish_oldest()

    def submit(self, payload, nbytes):
        """Apply backpressure before submitting an item; never retain GPU data."""
        if nbytes < 0 or nbytes > self.max_bytes:
            raise ValueError(f'score packet requires {nbytes} bytes; increase --effect_queue_mb '
                             f'(current limit {self.max_bytes} bytes)')
        self.check()
        while len(self._pending) >= self.max_items or self.pending_bytes + nbytes > self.max_bytes:
            self._finish_oldest()
        self._pending.append((self._executor.submit(self.consume, payload), nbytes))
        self.pending_bytes += nbytes
        self.peak_items = max(self.peak_items, len(self._pending))
        self.peak_bytes = max(self.peak_bytes, self.pending_bytes)

    def drain(self):
        """Finish every queued item, propagating failure to the caller."""
        while self._pending:
            self._finish_oldest()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                self.drain()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._pending.clear()
            self.pending_bytes = 0
