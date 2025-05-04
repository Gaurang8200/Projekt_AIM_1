import optuna
import copy
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from optuna.exceptions import TrialPruned


# ---------------------------- helper function ---------------------------- #

def evaluate_loss(trainer, loader):
    """Return average loss over a DataLoader (no gradients)."""
    trainer.model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(trainer.device), yb.to(trainer.device)
            loss = trainer.loss_fn(trainer.model(xb), yb)
            total += loss.item() * xb.size(0)
            count += xb.size(0)
    return total / count


# --------------------------- main tuning routine -------------------------- #

def run_optuna(model,
               train_subset,
               val_subset,
               TrainerClass,
               *,
               n_trials: int = 20,
               seed: int = 42):
    """
    Tune hyper‑parameters with Optuna on provided train/val subsets.

    Returns
    -------
    best_params : dict
    best_model_state : dict (state‑dict of best model)
    study : optuna.study.Study
    """
    initial_model = copy.deepcopy(model)  # untouched reference

    # ---- objective the sampler will call repeatedly ---- #
    def objective(trial):
        bs = trial.suggest_categorical("BS_SUGGEST", [32, 64, 128, 256])
        lr = trial.suggest_float("LR_SUGGEST", 1e-5, 5e-4, log=True)

        train_dl = DataLoader(train_subset, batch_size=bs, shuffle=True)
        val_dl   = DataLoader(val_subset,   batch_size=bs, shuffle=False)

        tuner = TrainerClass(model=copy.deepcopy(initial_model), lr=lr)

        best_val, best_state, no_improve = float("inf"), None, 0
        for epoch in range(1, 6):                                  # 5 epochs / trial
            tuner.train(train_dl, epochs=1, silent=True)
            val_loss = evaluate_loss(tuner, val_dl)
            trial.report(val_loss, epoch)

            if val_loss < best_val:
                best_val, best_state, no_improve = val_loss, \
                                                   copy.deepcopy(tuner.model.state_dict()), \
                                                   0
            else:
                no_improve += 1

            # prune if 2 consecutive epochs show no improvement
            if no_improve >= 2:
                raise TrialPruned()

        trial.set_user_attr("best_model_state", best_state)
        return best_val

    # ---- create and run study ---- #
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study   = optuna.create_study(direction="minimize",
                                  sampler=sampler,
                                  pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_params       = study.best_params
    best_model_state  = study.best_trial.user_attrs["best_model_state"]
    return best_params, best_model_state, study
