import random
import torch
from torch.utils.data import IterableDataset

class ShuffleBufferDataset(IterableDataset):
    """
    IterableDataset wrapper that maintains a rolling buffer for non-blocking large-shuffle.

    Args:
        dataset: any Iterable or map-style Dataset.
        buffer_size: int, number of examples to keep in the shuffle buffer.
    """
    def __init__(self, dataset, buffer_size: int):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        it = iter(self.dataset)
        buffer = []
        # Initial fill
        try:
            for _ in range(self.buffer_size):
                buffer.append(next(it))
        except StopIteration:
            pass

        # Yield-random-and-refill
        while buffer:
            idx = random.randrange(len(buffer))
            yield buffer[idx]
            try:
                buffer[idx] = next(it)
            except StopIteration:
                buffer.pop(idx)
