import optuna
import copy
import torch
from torch.utils.data import DataLoader
from optuna.exceptions import TrialPruned

def evaluate_accuracy(trainer, loader):
    trainer.model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(trainer.device), yb.to(trainer.device)
            preds = trainer.model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0

def run_optuna(model,
               train_subset,
               val_subset,
               TrainerClass,
               *,
               n_trials: int = 50,
               seed: int = 1):
    initial_model = copy.deepcopy(model)

    def objective(trial):
        bs         = trial.suggest_categorical("BS_SUGGEST", [16, 32, 64, 128])
        lr         = trial.suggest_float("LR_SUGGEST", 1e-6, 1e-2, log=True)
        max_epochs = trial.suggest_int("EPOCHS", 10, 35)

        train_dl = DataLoader(train_subset, batch_size=bs, shuffle=True)
        val_dl   = DataLoader(val_subset,   batch_size=bs, shuffle=False)

        tuner = TrainerClass(model=copy.deepcopy(initial_model), lr=lr)

        best_acc, best_state, no_improve = 0.0, None, 0
        for epoch in range(1, max_epochs + 1):
            tuner.train(train_dl, epochs=1, silent=True)
            val_acc = evaluate_accuracy(tuner, val_dl)

            # clamp to [0,1] in case of numeric glitch
            val_acc = max(0.0, min(val_acc, 1.0))

            trial.report(val_acc, epoch)

            if val_acc > best_acc:
                best_acc, best_state, no_improve = (
                    val_acc,
                    copy.deepcopy(tuner.model.state_dict()),
                    0
                )
            else:
                no_improve += 1

            if no_improve >= 3:
                raise TrialPruned()

        trial.set_user_attr("best_model_state", best_state)
        return best_acc

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study   = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )
    study.optimize(objective, n_trials=n_trials)

    best_params      = study.best_params
    best_model_state = study.best_trial.user_attrs["best_model_state"]
    return best_params, best_model_state, study
