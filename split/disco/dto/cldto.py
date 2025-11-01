from dataclasses import dataclass
import torch
from dto.config import ConfigDto

@dataclass
class clDto:
    """Data Transfer Object for training batch data and configuration

      u_batch : uncertainty related to the cohesiviness of a pairs in a batch
      epoch : Current epoch number
      CongigDto : Configuration DTO 
    """
    #masks: Maskdto  # Using the imported Masks class directly
    #u_batch: Tensor  # Batch tensor data
    epoch: int  # Current epoch number
    config: ConfigDto  # Configuration DTO - using imported class directly

    @property
    def get_epoch(self) -> int:
        """Get the current epoch number"""
        return self.epoch
    @property
    def get_config(self) -> ConfigDto:
        """Get the configuration DTO"""
        return self.config
  
    def to_device(self, device: torch.device) -> 'clDto':
        """Move all tensors to the specified device"""
        return clDto(
            epoch=self.epoch,
            config=self.config.to(device)
        )