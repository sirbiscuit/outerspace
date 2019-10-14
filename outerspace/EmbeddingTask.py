from multiprocessing import Process, Semaphore, Queue
from threading import Thread
from traitlets import HasTraits, Enum


class EmbeddingWorker(Process):
    def __init__(self, queue, X, y, transformation_method, embedding_args):
        super().__init__()
        self.pause_lock = Semaphore(value=True)  # lock is free
        self.embedding_args = embedding_args
        self.X = X
        self.y = y
        self.transformation_method = transformation_method
        self.queue = queue

    def callback(self, command, iteration, payload):
        # pausing acquires pause_lock and the following code only runs if
        # pause_lock is free
        with self.pause_lock:
            self.queue.put((command, iteration, payload))

    def run(self):
        self.transformation_method(
            self.X, self.y, self.embedding_args, self.callback)

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
        while True:
            command, iteration, payload = self.queue.get()

            self.invoke_callbacks(command, iteration, payload)

            if command == 'stop':
                # process all items of the queue
                while not self.queue.empty():
                    self.queue.get()
                break


class EmbeddingTask(HasTraits):
    status = Enum(['idle', 'running', 'paused'], default_value='idle')

    def __init__(self, X, y, transformation_method):
        super().__init__()
        self.X = X
        self.y = y
        self.process = None
        self.callbacks = []
        self.thread = None
        self.transformation_method = transformation_method
        self.queue = Queue(maxsize=2)  # ... because of memory

    def start(self, **embedding_args):
        if self.status == 'idle':
            self.process = EmbeddingWorker(
                self.queue, self.X, self.y, self.transformation_method,
                embedding_args)
            self.thread = MonitoringWorker(self.queue, self.callbacks)
            self.process.start()
            self.thread.start()
            self.status = 'running'

    def stop(self):
        if self.status in ['running', 'paused']:
            # First: pause process in order to empty the queue.
            # Why? Queue might be in a strange state if we stop the process
            # and look at queue elements afterwards. I suspect that e.g.
            # embeddings (numpy arrays) lie in process memory and by
            # terminating the process, the actual object gets lost and
            # queue.get() fails silently. But I have no idea if this is true.
            # Nevertheless terminating the process first and the thread later
            # causes a never-ending queue.get() call.
            self.process.pause_lock.acquire()

            # terminate thread
            self.queue.put(('stop', 0, None))
            self.thread.join()

            # terminate process
            self.process.terminate()
            self.process.join()  # TODO: measure speed gain when dropping this

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
