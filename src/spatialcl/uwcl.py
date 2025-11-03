from typing import Literal
from utils import*
from spatialcl.clearn import compute_weights_from_uncertainty
from spatialcl._dto.mask import Maskdto
from spatialcl._dto.config import ConfigDto
from spatialcl._dto.cldto import clDto
from spatialcl._dto.image_label_dto import ImageLabelDTO
from spatialcl.uncertainty import co_cluster_uncertainty

class DenseContrastiveLoss:
    """Senior implementation of dense contrastive loss with uncertainty weighting."""
    
    def __init__(
        self,
        config_dto: ConfigDto,
        image_label_dto: ImageLabelDTO,
    ):
        # Load configuration from YAML
        self.image_label_dto = image_label_dto
        self.config_dto = config_dto
    
        
    def __call__(
        self,
        z: Tensor,
        epoch: int,
    ) -> Tensor:
        return self.forward(z, epoch)
    
    def forward(
        self,
        z: Tensor,
        epoch: int,
    ) -> Tensor:
        """
        Compute dense contrastive loss with uncertainty weighting.
        
        Args:
            z: Feature tensor of shape [B, D] or [B, V, D]
            labels: Class labels of shape [B] 
            img_ids: Image identifiers of shape [B]
            epoch: Current training epoch
            u_batch: Optional pre-computed uncertainty matrix. If None, 
                    uncertainty_fn will be used to compute it.
            
        Returns:
            Scalar loss tensor
        """
        self._validate_inputs(z)
        # Flatten and normalize features
        z_flat, labels_flat, img_ids_flat=self.image_label_dto.flatten_inputs(z)

        u_batch = co_cluster_uncertainty(z=z_flat,labels=labels_flat,img_ids=img_ids_flat)
        
        loss =  self._compute_loss_per_batch(
            z_flat,
           labels_flat, 
           img_ids_flat,
           u_batch,
           epoch
        )
        
        return loss
    
 
    def _validate_inputs(
        self,
        z: Tensor,
    ) -> None:
        """Validate input tensor shapes and properties."""
        z_flat, labels_flat, img_ids_flat = self.image_label_dto.flatten_inputs(z)
        
        assert z_flat.dim() in [2, 3], f"z must be 2D or 3D tensor, got {z.dim()}D"
        assert labels_flat.dim() == 1, f"labels must be 1D tensor, got {labels_flat.dim()}D"
        assert img_ids_flat.dim() == 1, f"img_ids must be 1D tensor, got {img_ids_flat.dim()}D"
        
        B = z_flat.size(0)
        assert labels_flat.size(0) == B, f"Labels batch size mismatch: z={B}, labels={labels_flat.size(0)}"
        assert img_ids_flat.size(0) == B, f"Image IDs batch size mismatch: z={B}, img_ids={img_ids_flat.size(0)}"
    
    
    def _compute_similarity_matrix(self, z_flat: Tensor) -> Tensor:
        """Compute pairwise cosine similarity matrix."""
        return torch.matmul(z_flat, z_flat.T) / self.config_dto.temperature
    
    def _compute_loss_per_batch(
        self,
        z_flat : Tensor,
        labels_flat: Tensor,
        img_ids_flat: Tensor,
        u_batch: Tensor,
        epoch: int,
    ) -> Tensor:
        """Compute all necessary boolean masks for loss computation."""
        device = labels_flat.device
        
        # Identity mask (self-similarity)
        eye_mask = torch.eye(labels_flat.size(0), dtype=torch.bool, device=device)
        
        # Class and image similarity masks
        same_class = labels_flat.unsqueeze(0) == labels_flat.unsqueeze(1)
        same_image = img_ids_flat.unsqueeze(0) == img_ids_flat.unsqueeze(1)
        same_class[eye_mask] = False
        
        # Position masks
        strong_pos_mask = same_class & same_image
        weak_pos_mask = same_class & ~same_image
        pos_mask = strong_pos_mask | weak_pos_mask
        neg_mask = ~same_class

        masks = Maskdto( strong_pos_mask=strong_pos_mask,weak_pos_mask= weak_pos_mask,neg_mask= neg_mask,pos_mask= pos_mask,eye_mask= eye_mask
                        )
        
        diff_img_weight = compute_weights_from_uncertainty(
            uncertainty=u_batch, epoch=epoch, T= self.config_dto.T,
            device= self.config_dto.device,
            eps = self.config_dto.eps
            
        )
       
        pos_weights = strong_pos_mask.float() + diff_img_weight * weak_pos_mask.float()
        
        pos_weights[~masks.pos_mask] = 0.0
    
        # Compute similarity matrix and masks
        sim_matrix = self._compute_similarity_matrix(z_flat)

        exp_sim = torch.exp(sim_matrix)
        numerator = exp_sim * pos_weights
        # Weighted negative term with numerical stability
        exp_neg = exp_sim * neg_mask.float()
        neg_weights = exp_neg / (exp_neg.sum(dim=1, keepdim=True) + self.config_dto.eps)
        neg_term = neg_weights * exp_sim*neg_mask.float() 
        
        # Denominator: positive + negative terms
        denominator = numerator + neg_term + self.config_dto.eps

        # Log-probabilities
         # Compute log(numerator / denominator) only over positive pairs
        log_prob = torch.log(numerator / denominator)
        loss_matrix = -log_prob * pos_mask.float()

        # Per-sample loss: sum over j in Pos(i) / |Pos(i)|
        num_positives = pos_mask.sum(dim=1)
        valid = num_positives > 0
        loss = loss_matrix.sum(dim=1)[valid] / (num_positives[valid]+  self.config_dto.eps)

        return loss.mean()
        
# Example usage with co_cluster_uncertainty function
def build_uwcl(
    z :  Tensor,
    epoch:int ,
    temperature: float = 0.1,
    T: int = 100,
    eps: float = 1e-12,
    device: Optional[torch.device] = None,
    img_ids: Tensor = None,
    labels: Tensor = None,
   
) -> DenseContrastiveLoss:
    """
    Factory function to create DenseContrastiveLoss with co-cluster uncertainty.
    
    Args:
        temperature: Temperature scaling parameter
        same_img_weight: Weight for same-image positives
        T: Scheduling parameter
        eps: Numerical stability constant
        device: Device for computation
        
    Returns:
        Configured DenseContrastiveLoss instance
    """
    
    dense_contrastive = DenseContrastiveLoss(
            ConfigDto(temperature=temperature,T = T,eps=eps,device =device),
            ImageLabelDTO(img_id= img_ids,label=labels)
        )
    return dense_contrastive(z, epoch)
        
__all__ = ["build_uwcl"]  # Only this function is public

