import torch


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.1, restore_best_weights=False, weight_save_fpath='best.pth'):
        """
        早停类
        
        Args:
            patience (int): 在多少个epoch损失没有改善后停止训练，默认10
            min_delta (float): 认为有改善的最小变化值，默认0
            restore_best_weights (bool): 是否在停止时恢复最佳权重，默认False
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None
        self.weight_save_fpath = weight_save_fpath
        
    def __call__(self, loss, model=None):
        """
        在每个epoch结束时调用
        
        Args:
            loss (float): 当前epoch的损失值
            model: 模型对象，用于保存最佳权重
        """
        if self.best_loss is None:
            # 第一次调用，初始化最佳损失
            self.best_loss = loss
            if self.restore_best_weights and model is not None:
                self.best_weights = model
                torch.save(self.best_weights, self.weight_save_fpath)
                print("Saved model weights from the first epoch.")
        elif loss < self.best_loss - self.min_delta:
            # 损失有改善
            self.best_loss = loss
            self.counter = 0
            if self.restore_best_weights and model is not None:
                self.best_weights = model
                torch.save(self.best_weights, self.weight_save_fpath)
                print("Saved model weights from the best epoch.")
        else:
            # 损失没有明显改善
            self.counter += 1
            print(f'EarlyStopping: {self.counter}/{self.patience} [previous best loss: {self.best_loss}]')

            if loss < self.best_loss: # 损失有较小改善时，更新最优权重，但是触发早停机制
                self.best_loss = loss
                if self.restore_best_weights and model is not None:
                    self.best_weights = model
                    torch.save(self.best_weights, self.weight_save_fpath)
                    print("Saved model weights from the best epoch.")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and model is not None and self.best_weights is not None:
                    torch.save(self.best_weights, self.weight_save_fpath)
                    print("Saved model weights from the best epoch.")
    
    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None