from openTSNE import TSNE
from multiprocessing import Process, Semaphore, Lock, Condition, RawArray, RawValue
import warnings
import numpy as np
import time
from threading import Thread


class EmbeddingWorker(Process):
    def __init__(self, X, is_paused=False, **embedding_args):
        super().__init__()
        # self.early_exaggeration_iter = 0
        self.iteration = RawValue('I')
        self.iteration.value = 0
        # self.exaggeration_phase = True
        self.pause_lock = Semaphore(value=not is_paused)
        self.render_lock = Lock()
        self.work_signal = Condition()
        self.embedding_args = embedding_args
        self.X = X
        self.current_embedding = RawArray('f', len(self.X)*2)

    def callback(self, i, error, embedding):
        # if user paused then this lock can not be acquired
        # (until user clicks play again)
        with self.pause_lock, self.render_lock:
            current_embedding_reshaped = np.frombuffer(self.current_embedding, 'f').reshape(embedding.shape)
            np.copyto(current_embedding_reshaped, embedding)

            self.iteration.value = i

#             if self.exaggeration_phase:
#                 self.iteration.value = i
#             else:
#                 self.iteration.value = i + self.early_exaggeration_iter

#             if self.exaggeration_phase and i == self.early_exaggeration_iter:
#                 self.exaggeration_phase = False

            with self.work_signal:
                self.work_signal.notify_all()

    def run(self):
        tsne = TSNE(**self.embedding_args,
                    min_grad_norm=0,  # never stop
                    callbacks=self.callback,
                    callbacks_every_iters=1)

#         self.early_exaggeration_iter = tsne.early_exaggeration_iter
#         self.exaggeration_phase = tsne.early_exaggeration_iter > 0

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r"\nThe keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.")
            tsne.fit(self.X)

    def pause(self):
        self.pause_lock.acquire()

    def resume(self):
        self.pause_lock.release()

    def is_paused(self):
        return not self.pause_lock.get_value()


class MonitoringWorker(Thread):
    def __init__(self, process, callbacks):
        super().__init__()
        self.process = process
        self.callbacks = callbacks

    def run(self):
        process = self.process
        last_time = time.time()
        while True:
            process.work_signal.acquire()
            process.work_signal.wait()
            with process.render_lock:
                # measure speed
                now = time.time()
                iteration_duration = now - last_time
                last_time = now

                # reshape current_embedding
                current_embedding = np.frombuffer(process.current_embedding, 'f').reshape((len(process.X), 2))

                for callback in self.callbacks:
                    callback(process.iteration.value, current_embedding, iteration_duration)


class EmbeddingTask:
    def __init__(self, X):
        super().__init__()
#         self.early_exaggeration_iter = 0
#         self.iteration = RawValue('I')
#         self.iteration.value = 0
#         self.exaggeration_phase = True
        self.X = X
        self.process = None
        self.callbacks = []
        self.thread = None
        self.is_running = False

    def start(self, is_paused=False, **embedding_args):
        self.is_running = True
        if self.process is None:
            self.process = EmbeddingWorker(self.X, is_paused, **embedding_args)
            self.thread = MonitoringWorker(self.process, self.callbacks)
        self.process.start()
        self.thread.start()

    def stop(self):
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join()
            self.process = None
        self.is_running = False

    def pause(self):
        self.process.pause_lock.acquire()

    def resume(self):
        if self.process is None:
            self.start()
        else:
            self.process.pause_lock.release()

    def is_paused(self):
        return not self.process.pause_lock.get_value()

    def is_running(self):
        return self.is_running

    def add_handler(self, callback):
        self.callbacks.append(callback)
