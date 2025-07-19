from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.history = []

    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analysera kontext och returnera insikter"""
        pass

    @abstractmethod
    def act(self, analysis: Dict[str, Any]) -> Any:
        """Utför handling baserat på analys"""
        pass

    def observe(self, result: Any) -> None:
        """Observera resultat av handling"""
        self.history.append(result)
