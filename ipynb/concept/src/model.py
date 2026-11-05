from collections import OrderedDict
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .dataset import ConceptBank


# From https://github.com/uzukidd/sl-cbm/blob/b0429c55178d98d592fd56bb43e5650236b1aa77/utils/model_utils.py#L111.
class ResNetBottom(nn.Module):
    def __init__(self, original_model: nn.Module):
        super(ResNetBottom, self).__init__()

        # Takes in `features` and `classifier` from `pytorchcv`'s definition
        # of ResNet, which only has two children
        self.features = list(original_model.children())[0]
        self.classifier: nn.Linear = list(original_model.children())[1]  # type: ignore

        self.final_pool = list(self.features.children())[-1]
        self.features = nn.Sequential(
            OrderedDict(list(self.features.named_children())[:-1])
        )

        # NOTE: self.output_visual_patches = True

    def forward(self, x):
        visual_patches = self.features(x)  # [B, D, H, W]
        embedding = self.final_pool(visual_patches)  # [B, D, 1, 1]
        embedding = torch.flatten(embedding, 1)  # [B, D]

        return embedding, visual_patches.permute((0, 2, 3, 1))


# From https://github.com/uzukidd/sl-cbm/blob/b0429c55178d98d592fd56bb43e5650236b1aa77/utils/model_utils.py#L360.
class SimPool(nn.Module):
    SIMPOOL_DEBUG_FLAG = False

    def __init__(
        self,
        dim,
        num_heads=1,
        qkv_bias=False,
        qk_scale=None,
        gamma=None,
        use_beta=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.norm_patches = nn.LayerNorm(dim, eps=1e-6)

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)

        if gamma is not None:
            self.gamma = torch.tensor([gamma])

            if use_beta:
                self.beta = nn.Parameter(torch.tensor([0.0]))

        self.eps = torch.tensor([1e-6])

        self.gamma = gamma
        self.use_beta = use_beta

    def prepare_input(self, x: torch.Tensor):
        if len(x.shape) == 3:  # Transformer
            # Input tensor dimensions:
            # x: (B, N, d), where B is batch size, N are patch tokens, d is depth (channels)
            B, N, d = x.shape
            gap_cls = x.mean(-2)  # (B, N, d) -> (B, d)
            gap_cls = gap_cls.unsqueeze(1)  # (B, d) -> (B, 1, d)

            return gap_cls, x
        if len(x.shape) == 4:  # CNN
            # Input tensor dimensions:
            # x: (B, H, W, d), where B is batch size, d is depth (channels), H is height, and W is width
            B, H, W, d = x.shape
            gap_cls = x.mean([1, 2])  # (B, H, W, d) -> (B, d)
            x = x.reshape(B, H * W, d)  # (B, d, H, W) -> (B, d, H*W) -> (B, H*W, d)
            gap_cls = gap_cls.unsqueeze(1)  # (B, d) -> (B, 1, d)

            return gap_cls, x
        else:
            raise ValueError(
                f"Unsupported number of dimensions in input tensor: {len(x.shape)}"
            )

    def forward(
        self,
        x: torch.Tensor,
        prepared_q: torch.Tensor | None = None,
        return_attn_map: bool = False,
    ):
        if self.SIMPOOL_DEBUG_FLAG:
            import pdb

            pdb.set_trace()

        # Prepare input tensor and perform GAP as initialization
        gap_cls, x = self.prepare_input(x)

        # Prepare queries (q), keys (k), and values (v)
        q, k, v = gap_cls, self.norm_patches(x), self.norm_patches(x)

        if prepared_q is not None:
            q = prepared_q

        # Extract dimensions after normalization
        Bq, Nq, dq = q.shape
        Bk, Nk, dk = k.shape
        Bv, Nv, dv = v.shape

        # Check dimension consistency across batches and channels
        # NOTE: modified for compatibility with ONNX exports, check whether
        # these assertions are for debugging or impl purposes
        if not torch.jit.is_tracing():  # type: ignore
            assert Bq == Bk == Bv
            assert dq == dk == dv

        # Apply linear transformation for queries and keys then reshape
        qq = (
            self.wq(q)
            .reshape(Bq, Nq, self.num_heads, dq // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # (Bq, Nq, dq) -> (B, num_heads, Nq, dq/num_heads)
        kk = (
            self.wk(k)
            .reshape(Bk, Nk, self.num_heads, dk // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # (Bk, Nk, dk) -> (B, num_heads, Nk, dk/num_heads)

        vv = v.reshape(Bv, Nv, self.num_heads, dv // self.num_heads).permute(
            0, 2, 1, 3
        )  # (Bv, Nv, dv) -> (B, num_heads, Nv, dv/num_heads)

        # Compute attention scores
        attn = (qq @ kk.transpose(-2, -1)) * self.scale
        # Apply softmax for normalization
        attn = attn.softmax(dim=-1)  # [B, 1, 1, _grid * _grid]

        # If gamma scaling is used
        if self.gamma is not None:
            # Apply gamma scaling on values and compute the weighted sum using attention scores
            x = torch.pow(
                attn @ torch.pow((vv - vv.min() + self.eps), self.gamma), 1 / self.gamma
            )  # (B, num_heads, Nv, dv/num_heads) -> (B, 1, 1, d)
            # If use_beta, add a learnable translation
            if self.use_beta:
                x = x + self.beta
        else:
            # Compute the weighted sum using attention scores
            x = (attn @ vv).transpose(1, 2).reshape(Bq, Nq, dq)
        if return_attn_map:
            return x.squeeze(1).squeeze(1), attn.squeeze(1).squeeze(1)

        return x.squeeze(1).squeeze(1)


# From https://github.com/uzukidd/sl-cbm/blob/b0429c55178d98d592fd56bb43e5650236b1aa77/utils/model_utils.py#L43.
def generate_topk_mask(W, nec=5):
    num_classes, num_concepts = W.shape
    mask = torch.zeros_like(W)

    for c in range(num_classes):
        weights = W[c].abs()
        topk_indices = torch.topk(weights, k=nec).indices
        mask[c, topk_indices] = 1.0

    return mask


# Adapted from the class `spss_scbm` in https://github.com/uzukidd/sl-cbm/blob/main/models/spss_cbm.py.
# Class is greatly simplified, defaults taken from https://github.com/uzukidd/sl-cbm/blob/main/scripts/CUB/spss_resnet_cbm/spss_resnet_cbm_TCAV_with_lambda.sh.
class SLCBM(nn.Module):
    TRAINABLE_COMPONENTS = [
        "simpool",
        "token_projection",
        "classifier",
        "concepts_projection",
        "nec_concepts_projection",
    ]

    IS_CONCEPT_PROBABILITY_SPACE = False

    def __init__(
        self,
        backbone: ResNetBottom,
        normalizer,
        concept_bank: ConceptBank,
        num_classes: int = 10,
        weights=None,
    ):
        super().__init__()

        self.output_class = False
        self.output_logit = True
        self.output_embedding = False
        self.output_concepts = False

        self.backbone = backbone
        self.normalizer = normalizer
        self.concept_bank = concept_bank

        # CAV: concept activation vector
        self.cavs: torch.Tensor = (
            self.concept_bank.vectors.detach().clone().to(device="mps")
        )  # [18, D]
        self.concept_names: list[str] = self.concept_bank.concept_names.copy()

        self.n_concepts = self.cavs.shape[0]
        self.num_concepts = len(self.concept_bank.concept_names)
        self.num_classes = num_classes

        # `args.pooling_type = "simpool"` by default
        # SimPool: attention-based pooling method for convolutional/transformer
        # frameworks
        # https://github.com/billpsomas/simpool/tree/master
        self.simpool = SimPool(
            self.num_concepts,
            num_heads=1,
            qkv_bias=False,
            qk_scale=None,
            use_beta=True,
        )

        self.embedding_size = token_width = self.backbone.classifier.in_features

        self.token_projection = nn.Linear(
            in_features=token_width, out_features=self.num_concepts
        )

        self.concepts_projection = nn.Linear(
            in_features=self.num_concepts, out_features=self.num_classes
        )

        # NOTE: consider whether to add back `NECLinear` from SL-CBM
        # self.nec_concepts_projection = NECLinear(self.num_of_concepts, self.num_of_classes, nec=5)

        # NOTE: since we are sigmoiding the concept layer, there is no
        # need for another normalisation layer as below:
        # self.classifier = nn.Sequential(
        #     nn.LayerNorm(self.num_concepts),
        #     self.concepts_projection
        # )
        self.classifier = self.concepts_projection

        for name, param in self.named_parameters():
            if name.split(".")[0] in self.TRAINABLE_COMPONENTS:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def output_type(self, type: str):
        self.output_class = False
        self.output_logit = False
        self.output_embedding = False
        self.output_concepts = False

        setattr(self, f"output_{type}", True)

    # ------------
    # Getter & Setter
    # ------------
    def get_num_classes(self):
        return self.num_classes

    def get_num_concepts(self):
        return self.n_concepts

    def train(self, mode=True) -> nn.Module:
        for p in self.backbone.parameters():
            p.requires_grad = False

        return super().train(mode)

    def set_weights(self, weights: torch.Tensor, bias: torch.Tensor):
        self.classifier.weight.data = torch.tensor(weights).to(
            device=self.classifier.weight.device  # type: ignore
        )
        self.classifier.bias.data = torch.tensor(bias).to(
            device=self.classifier.weight.device  # type: ignore
        )
        return True

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        return {
            k: v
            for k, v in super().state_dict().items()
            if k.split(".")[0] in self.TRAINABLE_COMPONENTS
        }

    def forward(
        self, input_X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(input_X.size()) == 5:
            B, N, C, W, H = input_X.size()
            images = torch.reshape(input_X, (B * N, C, W, H))
        else:
            images = input_X

        pooled_tokens, token_concepts = self.encode_as_concepts(
            images, return_token_concepts=True
        )

        #     (bs*2,C)         (bs*2,Class)
        return self.classifier(pooled_tokens), pooled_tokens, token_concepts

    def encode_as_concepts(
        self,
        batch_X: torch.Tensor,
        return_token_concepts: bool = False,
        return_attn_map: bool = False,
    ) -> torch.Tensor:
        B, C, H, W = batch_X.size()
        images = self.normalizer(batch_X)

        image_embedding, visual_patches = self.backbone.encode_image(images)  # [B, D2]
        # H * W tokens + cls token x D1, cls token x D1 -> proj -> D2 == Embedding
        # 1x1 convolution
        # NOTE: `token_concepts` are the individual concept saliency maps
        token_concepts = self.token_projection(
            visual_patches
        )  # [B, H * W, D1] -> [B, H * W, C]

        # [B, H * W, D] * [C, D]
        # NOTE: `self.concept_softmax = False` by default
        # token_concepts = F.softmax(token_concepts, dim=2)

        # prepare cavs as query
        # image_embedding @ self.cavs
        # cav_query = self.cavs.mean(1).unsqueeze(0).unsqueeze(0).expand((B, -1, -1)) # [C, D2]-> [B, 1, C]
        # cav_query = self.CAV_layer(image_embedding).unsqueeze(1) # [C, D2]-> [B, 1, C]
        # [B, D2] @ [C, D2].T -> [B, C] -> [B, 1, C]
        cav_query = (image_embedding @ self.cavs.T).unsqueeze(1)  # [C, D2]-> [B, 1, C]

        attn_map = None
        if return_attn_map:  # [B, H * W, C], [B, 1, C]
            pooled_tokens, attn_map = self.simpool(
                token_concepts, cav_query, return_attn_map=True
            )  # [B, C]
            _grid = int(np.round(np.sqrt(attn_map.size(-1))))
            attn_map = attn_map.view(B, 1, _grid, _grid)
        else:
            pooled_tokens = self.simpool(token_concepts, cav_query)  # [B, C]

        # NOTE: `pooled_tokens` are the "confidences" of each individual concept.
        # `F.sigmoid` is used to convert all values to be between [0..1], indicating
        # the "likelihood" of a concept existing
        pooled_tokens = F.sigmoid(pooled_tokens)

        res_tuple = (pooled_tokens,)
        if return_token_concepts:
            res_tuple = res_tuple + (token_concepts,)

        if return_attn_map:
            res_tuple = res_tuple + (attn_map,)

        if res_tuple.__len__() == 1:
            return res_tuple[0]

        return res_tuple

    def attribute_attn_map(self, batch_X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch_X: [B, C, W, H]
            target: int/[B, 1]
        Return:
            attribution: [B, 1, _grid, _grid]
        """
        # [C] [B, 1, grid, grid]
        _, attn_map = self.encode_as_concepts(batch_X, return_attn_map=True)
        attn_map = F.interpolate(attn_map, batch_X.size()[-2:], mode="bicubic")

        return attn_map

    def attribute(
        self,
        batch_X: torch.Tensor,
        target: Union[torch.Tensor, int],
        additional_args: dict = {},
    ) -> torch.Tensor:
        """
        Args:
            batch_X: [B, C, W, H]
            target: int/[B, 1]
        Return:
            attribution: [B, 1, _grid, _grid]
        """
        # [C] [B, grid * grid, C]
        _, token_concepts = self.encode_as_concepts(batch_X, return_token_concepts=True)

        if token_concepts.size().__len__() == 4:
            token_concepts = token_concepts.view(
                token_concepts.size(0), -1, token_concepts.size(-1)
            )
        B, N, C = token_concepts.size()

        if isinstance(target, torch.Tensor):
            # [B, grid * grid, 1]
            expanded_target = (
                target.unsqueeze(1).unsqueeze(1).expand(-1, N, -1)
            )  # [B, N, 1]
            attribution = token_concepts.gather(
                dim=2, index=expanded_target
            )  # [B, grid * grid, 1]
        else:
            # [1, grid * grid, C]
            attribution = token_concepts[
                :, :, target : target + 1
            ]  # [1, grid * grid, 1]

        # [B, grid * grid, 1] -> # [B, 1, grid, grid]
        _grid = int(np.round(np.sqrt(N)))
        attribution = attribution.permute((0, 2, 1)).view(B, 1, _grid, _grid)
        return attribution

    def get_topK_concepts(self, K: int = 5):
        """
        Args:
            K:int

        Returns:
            top_indices:[C, K]
            top_values:[C, K]
        """
        # (out_features, in_features)
        weight: torch.Tensor = self.classifier[1].weight.data  # type: ignore
        top_values, top_indices = torch.topk(weight, k=K, dim=1)
        top_values = top_values / torch.sum(top_values, dim=1, keepdim=True)

        return top_indices, top_values

    def forward_projs(self, concept_projs: torch.Tensor) -> torch.Tensor:
        return self.classifier(concept_projs)


class ConceptOnlySLCBM(nn.Module):
    """
    A wrapper around the SL-CBM model, which only outputs concept probabilities.
    """

    def __init__(self, model: SLCBM):
        super().__init__()
        self.model = model

    def forward(self, input_X: torch.Tensor) -> torch.Tensor:
        _class_probs, concept_probs, _concept_saliency_maps = self.model(input_X)
        return concept_probs


class WrappedSLCBM(nn.Module):
    """
    A wrapper around the SL-CBM model, which does not output concepts or their saliency maps.
    """

    def __init__(self, model: SLCBM):
        super().__init__()
        self.model = model

    def forward(self, input_X: torch.Tensor) -> torch.Tensor:
        class_probs, _concept_probs, _concept_saliency_maps = self.model(input_X)
        return class_probs
