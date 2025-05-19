from typing import Mapping, Optional, Iterator, Any
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime as dt
from tqdm.notebook import tqdm

from .dataset import *
args = {
    "fp16" : False,
    "profiler" : True,
    "gradAcc" : False,
    #"gradAccIter": 4
}

class Trainer(nn.Module):
    """Class for model training.

    Args:
        model (nn.Module): The neural network model to be trained.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
        optimizer (torch.optim.Optimizer, optional): Optimizer for updating the model parameters.
            If None, Adam optimizer is used with the specified learning rate. Defaults to None.
        loss_fn (nn.Module, optional): Loss function to be used during training.
            Defaults to nn.CrossEntropyLoss().
        device (torch.device, optional): Device on which to train the model ("cuda" or "cpu").
            Defaults to cuda if available else cpu.
    """

    def __init__(self,
                 model: nn.Module,
                 *,
                 lr: Optional[float] = 1e-4,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 loss_fn: nn.Module = nn.CrossEntropyLoss(),
                 device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
                 ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr) if optimizer is None else optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, dataloader: DataLoader, *, epochs: int = 100, silent: bool = False) -> None:
        """Trains the model for a specified number of epochs.

        Args:
            dataloader (DataLoader): DataLoader providing input data and targets.
            epochs (int, optional): Number of training epochs. Defaults to 100.
            silent (bool, optional): If True, suppresses printing loss during training. Defaults to False.
        """
        for _ in self.train_iter(dataloader, epochs=epochs, silent=silent):
            pass

    def train_iter(self, dataloader: DataLoader, *, epochs: int = 100, silent: bool = False) -> Iterator[nn.Module]:
        """Trains the model for a specified number of epochs and yields the model after each epoch.

        Args:
            dataloader (DataLoader): DataLoader providing input data and targets.
            epochs (int, optional): Number of training epochs. Defaults to 100.
            silent (bool, optional): If True, suppresses printing loss during training. Defaults to False.

        Yields:
            nn.Module: The trained model after each epoch.
        """
        model = self.model.to(self.device)
        self._optimizer_to(self.optimizer, self.device)

        if args["fp16"]:
            scaler = torch.amp.GradScaler()
        
        if args["profiler"]:
            prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            #schedule=torch.profiler.schedule(wait=2, warmup=3, active=2, repeat=1),
            schedule=torch.profiler.schedule(wait=0, warmup=3, active=100, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/trainer_profile'),
            record_shapes=True,
            profile_memory=True,
            # with_stack=True,
            # with_flops=True,
            # with_modules=True
            )
            prof.start()
        
        for epoch in tqdm(range(epochs)):
            model.train()
            for i, (data, target) in enumerate(dataloader):
                self.optimizer.zero_grad()
                x = data.to(self.device)
                y = target.to(self.device)
                
                if args["fp16"]:
                    with torch.amp.autocast(device_type="cuda"):
                        x = model(x)
                else:
                    x = model(x)
                if args["gradAcc"]:
                    loss = self.loss_fn(x,y)/args["gradAccIter"]
                else:
                    loss = self.loss_fn(x, y)

                if args["gradAcc"]:
                    if args["fp16"]:
                        scaler.scale(loss).backward()
                        if ((i + 1) % args["gradAccIter"] == 0) or (i + 1 == len(dataloader)):
                            scaler.step(self.optimizer)
                            scaler.update()
                            self.optimizer.zero_grad()
                    else:
                        loss.backward()
                        if ((i + 1) % args["gradAccIter"] == 0) or (i + 1 == len(dataloader)):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                else:
                    if args["fp16"]:
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                if args["profiler"]:
                    prof.step()
                    
            yield model
        
        if args["profiler"]:
            prof.stop()

    def _optimizer_to(self, optim: torch.optim.Optimizer, device: torch.device) -> None:
        """Moves the optimizer's state to the specified device.

        Args:
            optim (torch.optim.Optimizer): The optimizer whose state needs to be moved.
            device (torch.device): The target device ("cuda" or "cpu").
        """
        for param in optim.state.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def state_dict(self) -> dict[str, Any]:
        """Returns the state dictionary of the trainer including the model's state and optimizer's state.

        Returns:
            dict[str, Any]: State dictionary containing both model and optimizer states.
        """
        sd = super().state_dict()
        sd["optimizer"] = self.optimizer.state_dict()
        return sd

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        """Loads the trainer's state from a provided state dictionary.

        Args:
            state_dict (Mapping[str, Any]): State dictionary containing model and optimizer states.
            strict (bool, optional): Whether to enforce that all keys in the state_dict match
                the keys returned by this module's `state_dict()`. Defaults to True.
            assign (bool, optional): If True, assigns the loaded parameters directly to the model.
                Defaults to False.
        """
        self.optimizer.load_state_dict(state_dict["optimizer"])
        del state_dict["optimizer"]
        super().load_state_dict(state_dict, strict, assign)
