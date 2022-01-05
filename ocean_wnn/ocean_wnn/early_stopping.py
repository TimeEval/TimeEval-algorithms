from typing import Iterator, Optional, List, Callable


class EarlyStopping:
    def __init__(self, patience: int, delta: float, epochs: int,
                 callbacks: Optional[List[Callable[[bool, float, int], None]]] = None):
        self.patience = patience
        self.delta = delta
        self.epochs = epochs
        self.current_epoch = 0
        self.epochs_without_change = 0
        self.last_loss: Optional[float] = None
        self.callbacks = callbacks or []

    def _callback(self, improvement: bool, loss: float):
        for cb in self.callbacks:
            cb(improvement, loss, self.epochs_without_change)

    def update(self, loss: float):
        improvement = False
        if self.last_loss is None or (1 - (loss / self.last_loss) > self.delta):
            self.last_loss = loss
            self.epochs_without_change = 0
            improvement = True
        else:
            self.epochs_without_change += 1

        self._callback(improvement, loss)

    def __iter__(self) -> Iterator[int]:
        while self.epochs_without_change <= self.patience and self.current_epoch < self.epochs:
            yield self.current_epoch
            self.current_epoch += 1
