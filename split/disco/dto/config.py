from split.utils import*
from typing import Literal

class ConfigDto:
    """
    Data Transfer Object for configuration parameters and tensor dimensions.

    Attributes:
        temperature (Optional[float]): Temperature for scaling.
        same_img_weight (Optional[float]): Weight assigned to same-image pairs.
        T (Optional[int]): Total number of epochs.
        eps (Optional[float]): Small value for numerical stability.
        device (Optional[torch.device]): Target device ("cuda" or "cpu").
        method (Optional[str]): Method used to compute uncertainty weights (e.g., "exp" or "tanh").
"""

    def __init__(
        self,
        temperature: Optional[float] = None,
        same_img_weight: Optional[float] = None,
        T: Optional[int] = None,
        eps: Optional[float] = None,
        device: Optional[torch.device] = None,
        method: Literal["exp", "tanh"] = "exp",
    ):
        self.temperature = temperature
        self.same_img_weight = same_img_weight
        self.method =  method
        self.T = T
        self.eps = eps
        self.device = device

    def to(self, device: Optional[torch.device] = None) -> "ConfigDto":
        """
        Set the target device for this configuration.

        Args:
            device (torch.device, optional): Target device ("cuda" or "cpu").

        Returns:
            ConfigDto: A new instance with the same values and updated device.
        """
        target_device = device or self.device
        return ConfigDto(
            temperature=self.temperature,
            same_img_weight=self.same_img_weight,
            T=self.T,
            eps=self.eps,
            device=target_device,
            method = self.method
        )
