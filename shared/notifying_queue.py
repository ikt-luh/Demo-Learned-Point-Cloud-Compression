import threading
import queue

class NotifyingQueue:
    def __init__(self):
        self.queue = queue.Queue()
        self.condition = threading.Condition()

    def put(self, item):
        with self.condition:
            self.queue.put(item)
            self.condition.notify()  # Notify a waiting thread

    def get(self):
        with self.condition:
            while self.queue.empty():  # Avoid spurious wakeups
                self.condition.wait()  # Wait for notification
            return self.queue.get()
