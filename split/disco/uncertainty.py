from split.utils import*
from .dto.image_label_dto import ImageLabelDTO
 
class CoClusterUncertainty:
    """Compute masked uncertainty between embeddings using Subjective Logic."""

    def __init__(self, image_label_dto:ImageLabelDTO , prior_weight:int = 2):
        
        self.prior_weight = prior_weight
        self.image_label_dto = image_label_dto
      
    def __call__(self, z: torch.Tensor, image_label_dto:ImageLabelDTO) -> torch.Tensor:
        return self.compute(z, image_label_dto)
    
    def compute(self, z: torch.Tensor,) -> torch.Tensor:
        """Compute uncertainty matrix with label masking."""
        # Validate and flatten inputs
       
        Z_flat, labels_flat,_ = self.image_label_dto.flatten_inputs(z)
        
        # Compute similarity 
        Z_norm = F.normalize(Z_flat, dim=1)
        sim = torch.matmul(Z_norm, Z_norm.T)

        #Compute uncertainty
        uncertainty = self._similarity_to_evidence(sim)

        mask = labels_flat.unsqueeze(1) == labels_flat.unsqueeze(0)
        uncertainty[~mask] = 0.0  # or np.nan if you prefer
        return uncertainty
    
    def _similarity_to_evidence(self, sim: torch.Tensor) -> torch.Tensor:
        """Convert similarity to uncertainty using subjective logic."""
        g_sim = sim,
        g_dsim = 1.0 - sim
        
        e_pos = F.softmax(g_sim, dim=1)
        e_neg = F.softmax(g_dsim, dim=1)
        
        total_mass = torch.exp(e_pos) + torch.exp(e_neg) + self.prior_weight
        return self.prior_weight / total_mass
    
    
# Module-level functions
def co_cluster_uncertainty(
    z: torch.Tensor, 
    image_label_dto: ImageLabelDTO,
    prior_weight : int = 2 
) -> torch.Tensor:
    """
        Direct function interface for co-cluster uncertainty.

        Args:
            Z (torch.Tensor): Embedding features tensor.
            labels (torch.Tensor, optional): Corresponding class labels for each embedding.
            img_ids (torch.Tensor, optional): Image identifiers for grouping instances.
            prior_weight (float): Prior weight indicating how many times an instance
                is assumed to be observed under a prior assumption.
            others : 

        Returns:
            torch.Tensor: Co-cluster uncertainty matrix.
        """
    computer = CoClusterUncertainty(image_label_dto=image_label_dto,prior_weight=prior_weight)
    return computer(z,image_label_dto)
