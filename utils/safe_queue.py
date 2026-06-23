# utils/safe_queue.py
"""
Thread‑safe queue for inter‑module communication.
Thin wrapper around queue.Queue with a few convenience methods.
"""

import queue


class SafeQueue:
    def __init__(self):
        self._queue = queue.Queue()

    def put(self, message):
        """Put a message dict into the queue."""
        self._queue.put(message)

    def get_nowait(self):
        """Get a message without blocking; raises queue.Empty if none."""
        return self._queue.get_nowait()

    def get(self, timeout=None):
        """Get a message, optionally blocking with timeout."""
        return self._queue.get(timeout=timeout)

    def empty(self):
        return self._queue.empty()

    def clear(self):
        """Remove all pending messages."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break