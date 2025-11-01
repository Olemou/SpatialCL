from split.utils import *
from config.disco_config import load_config
from typing import Dict
from dataclasses import field

@dataclass
class ImageLabelDTO:
    """
    Data Transfer Object for image identifiers and labels.

    Attributes:
        img_id (torch.Tensor): Tensor of image identifiers.
        label (torch.Tensor): Tensor of corresponding class labels.
    """
    img_id: torch.Tensor
    label: torch.Tensor
    device: Optional[torch.device] = None
    _B :int = field(default=64, init=False)
    _V :int = field(default=2, init=False)
    _D: int = field(default=128, init=False)
    _N: int = field(default=512, init=False)
    # Internal config storage (not in __init__)
    _config: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Load configuration after object initialization"""
        self._config = load_config()

    def to(self, device: Optional[torch.device] = None) -> "ImageLabelDTO":
        """
        Move tensors to the specified device.

        Args:
            device (torch.device, optional): Target device ("cuda" or "cpu").

        Returns:
            ImageLabelDTO: A new instance with tensors on the target device.
        """
        if device is None:
            return self
        return ImageLabelDTO(
            img_id=self.img_id.to(device),
            label=self.label.to(device)
        )
    def compute_dimensions(self, z: Tensor):
        """Compute and set dimensions based on input tensor shape"""
        if z.dim() == self._config["learning_mechanism"]["init_args"]["minimum_dim_output"]:
            self._B, self._D = z.shape
            self._V = self._config["learning_mechanism"]["init_args"]["number_patch_or_single_view"]
              # Single view per sample
            self._N = self._B
        else:  # dim == 3
            self._B, self.V, self._D = z.shape
            self._N = self._B * self._V
        
        return self
    def flatten_inputs(
        self,
        z: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Flatten and normalize input tensors using ImageLabelDTO.

        Args:
            z (Tensor): Input tensor of shape [B, D] or [B, V, D].
         
        Returns:
            Tuple[Tensor, Tensor, Tensor]: Flattened and normalized (z, labels, img_ids) on the target device.
        """
        # Move input and DTO to the target device
        if self.device is not None:
            z = z.to(self.device)
            self.label =self.label.to(self.device)
            self.img_id = self.img_id.to(self.device)

        self.compute_dimensions(z)

        if z.dim() == 2:
            z_flat = F.normalize(z, dim=1, eps=self.eps)
            labels_flat = self.label
            img_ids_flat = self.img_id
        else:
            B, V, D = self._B, self._V, self._D
            z_flat = F.normalize(z.view(B * V, D), dim=1, eps=self.eps)
            labels_flat = self.label.repeat_interleave(V)
            img_ids_flat = self.img_id.repeat_interleave(V)

        return z_flat, labels_flat, img_ids_flat
