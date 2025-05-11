import random
from torch.utils.data import IterableDataset

def evaluate_accuracy(trainer, loader, set_eval: bool = True) -> float:
    """
    Compute accuracy of trainer.model on data from loader.

    Args:
        trainer: object with .model and .device attributes.
        loader: DataLoader yielding (xb, yb) batches.
        set_eval: if True, sets model to eval mode before computing.

    Returns:
        Accuracy as a float (correct / total).
    """
    if set_eval:
        trainer.model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(trainer.device), yb.to(trainer.device)
            preds = trainer.model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0


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
