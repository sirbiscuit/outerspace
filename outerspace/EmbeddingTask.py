from openTSNE import TSNE
from multiprocessing import Process, Semaphore, Queue
import sharedmem
import warnings


class EmbeddingTask(Process):
    def __init__(self, X, is_paused=False, **embedding_args):
        super().__init__()
        self.early_exaggeration_iter = 0
        self.exaggeration_phase = True
        self.pause_lock = Semaphore(value=not is_paused)

        self.embedding_args = embedding_args

        self.X = X
        self.embedding = sharedmem.empty((len(X), 2))
        self.queue = Queue()

    def callback(self, i, error, embedding):
        self.embedding[:] = embedding

        # if user paused then this lock can not be aquired (until user clicks play again)
        with self.pause_lock:
            if not self.exaggeration_phase:
                i = i + self.early_exaggeration_iter

            self.queue.put(
                dict(
                    iteration=i,
                    exaggeration_phase=self.exaggeration_phase))

            if self.exaggeration_phase and i == self.early_exaggeration_iter:
                self.exaggeration_phase = False

    def run(self):
        tsne = TSNE(**self.embedding_args,
                    callbacks=self.callback,
                    callbacks_every_iters=1)

        self.early_exaggeration_iter = tsne.early_exaggeration_iter
        self.exaggeration_phase = tsne.early_exaggeration_iter > 0

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r"\nThe keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.")
            tsne.fit(self.X)

    def pause(self):
        self.pause_lock.acquire()

    def resume(self):
        self.pause_lock.release()

    def is_paused(self):
        return not self.pause_lock.get_value()
