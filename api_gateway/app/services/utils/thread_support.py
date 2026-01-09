import threading
from abc import ABC, abstractmethod


class ThreadSupportMixin(ABC):
    def __init__(self, *args, **kwargs):
        # Initialize the parent(s) in the MRO chain
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()
        self._thread = None
        if not hasattr(self, "refresh_interval"):
            self.refresh_interval = 60

    @abstractmethod
    def thread_mixin_target_function(self):
        """The child class MUST implement this."""
        pass

    def refresh_loop(self):
        while not self._stop_event.is_set():
            try:
                self.thread_mixin_target_function()
            except Exception as e:
                # Use the class name so you know WHICH service failed
                print(f"[{self.__class__.__name__}] Refresh failed: {e}")

            self._stop_event.wait(int(self.refresh_interval))

    def start(self):
        self._thread = threading.Thread(target=self.refresh_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
