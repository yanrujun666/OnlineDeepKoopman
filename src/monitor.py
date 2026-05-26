import torch


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.1, restore_best_weights=False, weight_save_fpath='best.pth'):
        """
        Args:
            patience: epochs without meaningful improvement before stopping
            min_delta: minimum loss decrease to count as improvement
            restore_best_weights: keep best checkpoint dict in memory (best_weights)
            weight_save_fpath: path to persist the best checkpoint on disk
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.weight_save_fpath = weight_save_fpath

        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None

    def _save_best(self, model):
        """Persist best checkpoint; optionally retain it in memory."""
        torch.save(model, self.weight_save_fpath)
        if self.restore_best_weights:
            self.best_weights = model

    def __call__(self, loss, model=None):
        if self.best_loss is None:
            self.best_loss = loss
            if model is not None:
                self._save_best(model)
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            if model is not None:
                self._save_best(model)
        else:
            self.counter += 1
            print(f'EarlyStopping: {self.counter}/{self.patience} [previous best loss: {self.best_loss}]')

            if loss < self.best_loss:
                self.best_loss = loss
                if model is not None:
                    self._save_best(model)

            if self.counter >= self.patience:
                self.early_stop = True
                if self.best_weights is not None:
                    torch.save(self.best_weights, self.weight_save_fpath)

    def reset(self):
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None
