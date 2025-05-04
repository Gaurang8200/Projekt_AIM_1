import torch
from typing import *
import plotly.graph_objs as go
import plotly.express as px
from torch import nn
from IPython.display import clear_output, display
from ipywidgets import BoundedIntText
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
    MulticlassConfusionMatrix,
    MulticlassStatScores,
)
from datetime import datetime
import torch.nn.utils.prune as prune



def ISO_time() -> str:
    """
    Returns the current time in ISO 8601 format.
    Returns:
        str: Current time in ISO 8601 format.
    """
    return datetime.now().isoformat()


def apply_pruning(model, amount):
    """
    Applies L1 unstructured pruning to the convolutional layers of the model.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and "classifier" not in name:
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model


class InferenceSession(nn.Module):
    """
    A class to handle inference sessions with a given model.

    Attributes:
        model (nn.Module): The neural network model for inference.
        batch_size (int): The batch size to use for a forward pass. (avoids out-of-memory errors)
        device (torch.device, optional): Device to run the model on. Defaults to GPU if available, otherwise CPU.
        pin_memory (torch.device, optional): Device to put the result on to. Defaults to True if GPU is available, otherwise False.
    """
    def __init__(self, model:nn.Module, *, batch_size:int=128, device:torch.device = "cuda" if torch.cuda.is_available() else "cpu", pin_memory:torch.device = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.pin_memory = pin_memory
        self.batch_size = batch_size

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Perform inference with the given model.

        This method moves input tensors to the specified device and performs inference.
        If the first argument has more than `batch_size` elements, it splits the data into chunks of size `batch_size`,
        processes each chunk separately, and concatenates the results.

        Args:
            *args: Positional arguments to be passed to the model's forward method.
            **kwargs: Keyword arguments to be passed to the model's forward method.

        Returns:
            torch.Tensor: The result of the model's forward pass, moved to the specified device if pin_memory is True.
        """
        args = [arg.to(self.device) if isinstance(arg, torch.Tensor) else arg for arg in args]

        self.model = self.model.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            if len(args[0]) > self.batch_size:
                split_args = zip(*[torch.split(a, self.batch_size) for a in args])
                results = map(lambda t: self.model(*t, **kwargs), split_args)
                return torch.cat(list(results)).to(self.pin_memory)
            else:
                result = self.model(*args, **kwargs)
                return result.to(self.pin_memory)



class Plotter:
    @staticmethod
    def plot_dataset_item(item: tuple[torch.Tensor, torch.Tensor], *, title: str = "Dataset item", height: int = 300) -> go.Figure:
        """
        Plots a dataset item.

        Args:
            item (tuple[torch.Tensor, torch.Tensor]): The dataset item to be plotted. It should be a tuple
                containing the image tensor and the label.
            title (str, optional): The title of the plot. Defaults to "Dataset item".
            height (int, optional): The height of the plot in pixels. Defaults to 300.

        Returns:
            go.Figure: A Plotly Figure object representing the plot.
        """

        image = item[0]
        label = item[1]

        fig = px.imshow(image.permute(1, 2, 0))
        fig.update_layout(
            title=title + f"<br><sup>Label: {label}</sup>",
            xaxis_title="pulse",
            yaxis_title="value (us)",
            height=height
        )

        return fig

    @staticmethod
    def plot_sequential_data(item: torch.Tensor, *, ticks: Optional[torch.Tensor] = None, names: List[str] = None, title: str = "Sequential Data", xaxis_title: str = "x", yaxis_title: str = "y", height: int = 300, logarithmic: bool = False) -> go.Figure:
        """
        Plots sequential data with optional ticks and custom names.

        Args:
            item (torch.Tensor): The sequential data to be plotted.
            ticks (Optional[torch.Tensor], optional): Ticks for the x-axis. Defaults to None.
            names (List[str], optional): Names for each trace in the plot. Defaults to None.
            title (str, optional): The title of the plot. Defaults to "Sequential Data".
            xaxis_title (str, optional): Title for the x-axis. Defaults to "x".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "y".
            height (int, optional): The height of the plot in pixels. Defaults to 300.
            logarithmic (bool, optional): Whether to use a logarithmic scale on the y-axis. Defaults to False.

        Returns:
            go.Figure: A Plotly Figure object representing the plot.
        """
        fig = go.Figure()
        for i, element in enumerate(item):
            fig.add_trace(go.Scatter(y=element.cpu().numpy(), x=ticks.cpu().numpy(), mode="lines", name=f"{i}" if names is None else names[i]))
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            height=height,
            yaxis=dict(type="log" if logarithmic else "linear")
        )
        return fig

    @staticmethod
    def plot_confusion_matrix(cm: torch.Tensor, *, title: str = "Confusion Matrix", height: int = 300) -> go.Figure:
        """
        Plots a confusion matrix.

        Args:
            cm (torch.Tensor): The confusion matrix to be plotted.
            title (str, optional): The title of the plot. Defaults to "Confusion Matrix".
            height (int, optional): The height of the plot in pixels. Defaults to 300.

        Returns:
            go.Figure: A Plotly Figure object representing the confusion matrix.
        """
        cm = cm.cpu().numpy()
        fig = px.imshow(cm, labels=dict(x="Predicted Class", y="True Class", color="Number of Samples"), title=title)
        fig.update_layout(height=height)
        return fig



class NotebookPlotter(Plotter):
    """
    A utility class for plotting dataset items interactively in a Jupyter Notebook.
    """

    @staticmethod
    def plot_dataset_item_interactive(dataset: Any, *, title: str = "Dataset item", height: int = 300) -> None:
        """
        Plots an interactive widget to select and display a dataset item from a dataset.

        Args:
            dataset (DatasetBase): The dataset containing items to plot.
            title (str): Title of the plot. Defaults to "Dataset item".
            height (int): Height of the plot in pixels. Defaults to 300.

        Returns:
            None
        """
        w = BoundedIntText(value=0, min=0, max=len(dataset)-1, step=1, description='item:', disabled=False)
        
        def update(change):
            fig = NotebookPlotter.plot_dataset_item(dataset[w.value], title=title, height=height)
            clear_output(wait=True)
            display(w)
            display(fig)

        w.observe(update, "value")
        display(w)
        display(NotebookPlotter.plot_dataset_item(dataset[0], title=title, height=height))

    @staticmethod
    def plot_sequential_data_interactive(item: torch.Tensor, *, title: str = "Sequential Data", xaxis_title: str = "x", yaxis_title: str = "y", height: int = 300) -> None:
        """
        Plots an interactive widget to display sequential data.

        Args:
            item (torch.Tensor): The sequential data to plot.
            title (str): Title of the plot. Defaults to "Sequential Data".
            xaxis_title (str): Title of the x-axis. Defaults to "x".
            yaxis_title (str): Title of the y-axis. Defaults to "y".
            height (int): Height of the plot in pixels. Defaults to 300.

        Returns:
            None
        """
        fig = NotebookPlotter.plot_sequential_data(item, title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title, height=height)
        clear_output(wait=True)
        display(fig)



class Evaluator:
    @staticmethod
    def acc(data: torch.Tensor, target: torch.Tensor, *, topk: int = 1) -> torch.Tensor:
        """Compute accuracy for multiclass classification.

        Args:
            data (torch.Tensor): Model predictions.
            target (torch.Tensor): True labels.
            topk (int): Number of highest probability values to consider. Defaults to 1.

        Returns:
            torch.Tensor: Accuracy score.
        """
        target = target.to(data.device)
        n_classes = len(target.flatten().unique())
        return MulticlassAccuracy(num_classes=n_classes, top_k=topk).to(data.device)(data, target)

    @staticmethod
    def ppv(data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute precision (positive predictive value) for multiclass classification.

        Args:
            data (torch.Tensor): Model predictions.
            target (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Precision score.
        """
        target = target.to(data.device)
        n_classes = len(target.flatten().unique())
        return MulticlassPrecision(num_classes=n_classes).to(data.device)(data, target)

    @staticmethod
    def tpr(data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute recall (true positive rate) for multiclass classification.

        Args:
            data (torch.Tensor): Model predictions.
            target (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Recall score.
        """
        target = target.to(data.device)
        n_classes = len(target.flatten().unique())
        return MulticlassRecall(num_classes=n_classes).to(data.device)(data, target)

    @staticmethod
    def f1_score(data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute F1 score for multiclass classification.

        Args:
            data (torch.Tensor): Model predictions.
            target (torch.Tensor): True labels.

        Returns:
            torch.Tensor: F1 score.
        """
        target = target.to(data.device)
        n_classes = len(target.flatten().unique())
        return MulticlassF1Score(num_classes=n_classes).to(data.device)(data, target)

    @staticmethod
    def mcc(data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Matthews correlation coefficient (MCC) for multiclass classification.

        Args:
            data (torch.Tensor): Model predictions.
            target (torch.Tensor): True labels.

        Returns:
            torch.Tensor: MCC score.
        """
        target = target.to(data.device)
        n_classes = len(target.flatten().unique())
        return MulticlassMatthewsCorrCoef(num_classes=n_classes).to(data.device)(data, target)

    @staticmethod
    def cm(data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute confusion matrix for multiclass classification.

        Args:
            data (torch.Tensor): Model predictions.
            target (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Confusion matrix.
        """
        target = target.to(data.device)
        n_classes = len(target.flatten().unique())
        return MulticlassConfusionMatrix(num_classes=n_classes).to(data.device)(data, target)

    @staticmethod
    def stats(data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute various statistics for multiclass classification.

        This function returns a tensor containing the following metrics for each class:
            - True Positives (TP)
            - False Positives (FP)
            - True Negatives (TN)
            - False Negatives (FN)
            - Accuracy (acc)
            - Precision (ppv)
            - Recall (tpr)
            - Fall-Out Rate (fpr)
            - Miss Rate (fnr)
            - Specificity (tnr)
            - F1 Score
            - Matthews Correlation Coefficient (mcc)

        Args:
            data (torch.Tensor): Model predictions.
            target (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Statistics summary tensor of shape [n_classes, 12].
        """
        target = target.to(data.device)
        n_classes = len(target.flatten().unique())
        res = MulticlassStatScores(num_classes=n_classes).to(data.device)(data, target)
        tp, fp, tn, fn = res[:, 0], res[:, 1], res[:, 2], res[:, 3]
        acc = (tp + tn) / (tp + fp + tn + fn)
        ppv = tp / (tp + fp)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        tnr = tn / (tn + fp)
        f1 = 2 * (ppv * tpr) / (ppv + tpr)
        mcc = (tp * tn - fp * fn) / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        metrics = torch.stack([tp, fp, tn, fn, acc, ppv, tpr, fpr, fnr, tnr, f1, mcc], dim=0)
        return torch.nan_to_num(metrics, nan=0.0)