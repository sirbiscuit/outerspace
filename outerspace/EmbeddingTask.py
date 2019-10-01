from multiprocessing import Process, Semaphore, Queue
import time
from threading import Thread
from traitlets import HasTraits, Enum


class EmbeddingWorker(Process):
    def __init__(self, queue, X, transformation_method, **embedding_args):
        super().__init__()
        self.pause_lock = Semaphore(value=True)  # lock is free
        self.embedding_args = embedding_args
        self.X = X
        self.transformation_method = transformation_method
        self.queue = queue

    def callback(self, command, iteration, payload):
        # pausing acquires pause_lock and the following code only runs if
        # pause_lock is free
        with self.pause_lock:
            self.queue.put((command, iteration, payload))

    def run(self):
        self.transformation_method(
            self.X, None, self.embedding_args, self.callback)

    def pause(self):
        self.pause_lock.acquire()

    def resume(self):
        self.pause_lock.release()

    def is_paused(self):
        return not self.pause_lock.get_value()


class MonitoringWorker(Thread):
    def __init__(self, queue, callbacks):
        super().__init__()
        self.queue = queue
        self.callbacks = callbacks

    def invoke_callbacks(self, *args, **kwargs):
        for callback in self.callbacks:
            callback(*args, **kwargs)

    def run(self):
        last_time = time.time()
        last_iteration = 0
        while True:
            command, iteration, payload = self.queue.get()

            if command == 'stop':
                self.invoke_callbacks(command, iteration, payload)
                break

            # measure speed
            if iteration > last_iteration:
                now = time.time()
                iteration_duration = now - last_time
                last_time = now
                self.invoke_callbacks('speed', iteration, iteration_duration)
            last_iteration = iteration

            self.invoke_callbacks(command, iteration, payload)


class EmbeddingTask(HasTraits):
    status = Enum(['idle', 'running', 'paused'], default_value='idle')

    def __init__(self, X, transformation_method):
        super().__init__()
        self.X = X
        self.process = None
        self.callbacks = []
        self.thread = None
        self.transformation_method = transformation_method
        self.queue = Queue(maxsize=2)

    def start(self, **embedding_args):
        if self.status == 'idle':
            self.process = EmbeddingWorker(
                self.queue, self.X, self.transformation_method,
                **embedding_args)
            self.thread = MonitoringWorker(self.queue, self.callbacks)
            self.process.start()
            self.thread.start()
            self.status = 'running'

    def stop(self):
        if self.status in ['running', 'paused']:
            # terminate process
            self.process.terminate()
            self.process.join()
            self.process = None

            # terminate thread
            self.queue.put(('stop', 0, None))
            self.thread.join()
            self.thread = None

            self.status = 'idle'

    def pause(self):
        if self.status == 'running':
            self.process.pause_lock.acquire()
            self.status = 'paused'

    def resume(self):
        if self.status == 'paused':
            self.process.pause_lock.release()
            self.status = 'running'

    def is_paused(self):
        return not self.process.pause_lock.get_value()

    def add_handler(self, callback):
        self.callbacks.append(callback)
