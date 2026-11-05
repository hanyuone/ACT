from torch import Tensor
import torch

from act.front_end.specs import InKind, InputSpec


class ConceptInputSpec(InputSpec):
    @staticmethod
    def invert_mask(mask: Tensor) -> Tensor:
        return 1 - mask
    
    def __init__(self, image: Tensor, saliency_map: Tensor, eps: float):
        """
        ### Params
        - `image`: The image we want to generate an input spec for
        - `saliency_map`: Saliency map of the image for a specific concept, determines
          where we perturb
        - `eps`: The variation in possible perturbations
        """
        self.center = image
        self.eps = eps

        assert image.shape == saliency_map.shape

        change = saliency_map * eps
        self.lb = torch.clamp(image - change, 0.0, 1.0)
        self.ub = torch.clamp(image + change, 0.0, 1.0)

        super().__init__(
            kind=InKind.BOX,
            lb=self.lb,
            ub=self.ub,
            center=self.center,
            eps=self.eps
        )
