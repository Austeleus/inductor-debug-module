from abc import ABC, abstractmethod
import torch.fx

class Constraint(ABC):
    """
    Abstract base class for defining constraints for the Mock Backend.
    """
    
    @abstractmethod
    def check(self, node: torch.fx.Node) -> bool:
        """
        Checks if a single node satisfies the constraint.
        Returns True if valid, False if invalid.
        """
        pass

    @abstractmethod
    def message(self, node: torch.fx.Node) -> str:
        """
        Returns the error message if the check fails.
        """
        pass
