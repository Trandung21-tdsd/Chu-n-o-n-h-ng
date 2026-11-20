"""Early stopping theo metric validation + theo dõi thời điểm đạt chất lượng mục tiêu Q* (cho TATO)."""
from __future__ import annotations

import math


class EarlyStopping:
    """Dừng khi metric validation không cải thiện quá `min_delta` trong `patience` epoch liên tiếp."""

    def __init__(self, patience: int = 8, min_delta: float = 1e-3) -> None:
        if patience < 1:
            raise ValueError("patience phải >= 1")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = -math.inf
        self.best_epoch = 0
        self.bad_epochs = 0

    def step(self, value: float, epoch: int) -> bool:
        """Trả về True nếu epoch này là tốt nhất (cần lưu checkpoint)."""
        if not math.isfinite(value):
            value = -math.inf
        if value > self.best + self.min_delta:
            self.best, self.best_epoch, self.bad_epochs = float(value), int(epoch), 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


class TargetTracker:
    """Đạt mục tiêu khi metric >= Q* trong `consecutive` lần validation liên tiếp (định nghĩa TATO v2, mục 13)."""

    def __init__(self, target: float = 0.85, consecutive: int = 2) -> None:
        if consecutive < 1:
            raise ValueError("consecutive phải >= 1")
        self.target = float(target)
        self.consecutive = int(consecutive)
        self.streak = 0
        self.reached_epoch: int | None = None

    def step(self, value: float, epoch: int) -> bool:
        """Trả về True đúng 1 lần — ở epoch xác nhận đạt mục tiêu."""
        self.streak = self.streak + 1 if (math.isfinite(value) and value >= self.target) else 0
        if self.reached_epoch is None and self.streak >= self.consecutive:
            self.reached_epoch = int(epoch)
            return True
        return False
