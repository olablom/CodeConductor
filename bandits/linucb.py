import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


class LinUCBBandit:
    def __init__(self, d: int, alpha: float = 1.0):
        """
        d: dimensioner av feature-vektorn
        alpha: exploration parameter
        """
        self.d = d
        self.alpha = alpha
        self.A = defaultdict(lambda: np.eye(d))
        self.b = defaultdict(lambda: np.zeros((d, 1)))

    def get_ucb(self, arm: str, x: np.ndarray) -> float:
        """Beräkna Upper Confidence Bound för en arm"""
        x = x.reshape(-1, 1)
        A_inv = np.linalg.inv(self.A[arm])
        theta = A_inv @ self.b[arm]
        p = float(theta.T @ x) + self.alpha * float(np.sqrt(x.T @ A_inv @ x))
        return p

    def select_arm(self, arms: List[str], x: np.ndarray) -> str:
        """Välj bästa arm baserat på UCB"""
        ucbs = {arm: self.get_ucb(arm, x) for arm in arms}
        return max(ucbs, key=ucbs.get)

    def update(self, arm: str, x: np.ndarray, reward: float) -> None:
        """Uppdatera modell med observerad reward"""
        x = x.reshape(-1, 1)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x
