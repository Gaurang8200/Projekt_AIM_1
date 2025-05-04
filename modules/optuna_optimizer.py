import optuna
#from modules.trainer import Trainer 
from mai_project1_optimization.modules.trainer import Trainer
import torch
from torch.utils.data import DataLoader

class OptunaTuner:
    def __init__(self, model_fn, train_dataset, val_dataset, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_fn = model_fn
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device

    def objective(self, trial):
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs = trial.suggest_int("epochs", 10, 30)

        model = self.model_fn()
        trainer = Trainer(model, lr=lr, device=self.device)

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        trainer.train(train_loader, epochs=epochs, silent=True)

        # Evaluate on validation set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        return acc

    def run(self, n_trials=20, seed=42):
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(self.objective, n_trials=n_trials)
        return study
