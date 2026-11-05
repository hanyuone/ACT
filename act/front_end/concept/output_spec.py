from typing import Any, Optional, Sequence, cast

from torch import Tensor
import torch

from act.front_end.specs import OutKind, OutputSpec


class ConceptOutputSpec(OutputSpec):
    @staticmethod
    def batch(out_specs: Sequence["ConceptOutputSpec"], out_kind: OutKind) -> OutputSpec:
        match out_kind:
            case OutKind.CONCEPT_CONFIDENCE:
                # Confidence spec is RANGE under the hood, pass to superclass's implementation
                for o in out_specs:
                    o.kind = OutKind.RANGE

                return OutputSpec.batch(out_specs, OutKind.RANGE)
            case OutKind.CONCEPT_TOP_N:
                out_specs = cast(Sequence[TopN], out_specs)

                num_elements = out_specs[0].num_elements
                assert all(o.num_elements == num_elements for o in out_specs), "Cannot batch top-N specs that have different numbers of elements"

                y_true: Tensor = torch.cat([o.y_true for o in out_specs])
                top_n = torch.cat([o.top_n for o in out_specs])

                return TopN(num_elements, y_true, top_n)
            case OutKind.CONCEPT_TOP_IN_CATEGORY:
                out_specs = cast(Sequence[TopInCategory], out_specs)

                num_elements = out_specs[0].num_elements
                assert all(o.num_elements == num_elements for o in out_specs), "Cannot batch top-N specs that have different numbers of elements"

                category_indices = out_specs[0].category_indices
                assert all(o.category_indices == category_indices for o in out_specs), "Cannot batch top-N specs that have different numbers of elements"

                top_elements = torch.cat([o.top_elements for o in out_specs])
                is_top = torch.cat([o.is_top for o in out_specs])

                return TopInCategory(num_elements, category_indices, top_elements, is_top)
            case _: raise ValueError("OutKind must be a concept spec kind to be batched into ConceptOutputSpec")


class Confidence(ConceptOutputSpec):
    """
    Defines a specification that only "passes" when certain concepts we specify exist,
    with existence defined by a confidence value between 0 and 1.
    """
    concepts: Tensor

    def __init__(
        self,
        num_elements: int,
        y_true: Tensor,
        threshold: Tensor,
    ):
        """
        ### Params
        - `num_elements`: The number of concepts.
        - `y_true` [B]: The concepts we want to be affected by thresholds.
        - `threshold` [B]: The confidence thresholds of each individual concept.
        """
        if not (0 <= threshold <= 1):
            raise ValueError("Confidence threshold must be in [0, 1]")

        # `Confidence` is an `OutKind.RANGE`
        lb = torch.zeros((num_elements))
        lb[cast(int, y_true.item())] = threshold

        ub = torch.ones((num_elements))

        super().__init__(OutKind.CONCEPT_CONFIDENCE, lb=lb, ub=ub)


class TopN(ConceptOutputSpec):
    def __init__(self, num_elements: int, y_true: Tensor, threshold: Tensor):
        """
        ### Params
        - `num_elements`: The number of concepts.
        - `y_true` [B]: The "correct" concepts we want to be in the top-N.
        - `threshold` [B]: The rank we want our correct concepts to be within.
        """
        if (threshold > num_elements).any():
            raise ValueError("`top_n` must be smaller than number of elements")

        self.num_elements = num_elements
        # Specs can be *batched*
        self.y_true = y_true
        self.top_n = threshold

        super().__init__(OutKind.CONCEPT_TOP_N, y_true=self.y_true)

    def severity(self, y: Tensor, rows: Optional[Tensor] = None) -> Tensor:
        """
        Formula for the severity of a `TopRanked` violation:

        `s = [y_i forall i if i != t].sort()[n] - y_t`

        This is a temporary metric based on the existing Carlini-Wagner metric
        used for TOP1_ROBUST. It might need to be replaced once something that
        works better for multiclass classification is found in literature.
        """
        # `rows`: indices of the batched model's input that each tensor in output `y` corresponds to.
        # If `rows == None`, `rows[i] = i`.
        assert self.y_true is not None

        if rows == None:
            y_true = self.y_true
            top_n = self.top_n
        else:
            y_true = self.y_true[rows]
            top_n = self.top_n[rows]

        # 1. Get `top_n`-th item from `y`
        sorted = y.sort(dim=1, descending=True)[0].gather(dim=1, index=top_n.unsqueeze(dim=1))

        # 2. Calculate difference
        y_true_confidences = y.gather(dim=1, index=y_true.unsqueeze(dim=1))
        return (sorted - y_true_confidences).squeeze(dim=1)


class TopInCategory(ConceptOutputSpec):
    def __init__(
            self,
            num_elements: int,
            category_indices: list[list[int]],
            top_elements: Tensor,
            is_top: Tensor,
        ):
        """
        ### Params
        - `num_elements`: The number of concepts in the dataset.
        - `category_indices`: A collection of lists, with each inner list representing all concepts
          to do with a certain category
        - `top_elements` [B, Cat]: An int tensor where each individual tensor contains
          the indices of the top concept for each category.
        - `is_top` [B, Cat]: A bool tensor where each individual tensor contains which
          categories are actually counted as the top.
        """
        assert len(category_indices) == top_elements.shape[1], "Each category must have a top element"

        self.num_elements = num_elements
        self.category_indices = category_indices
        self.top_elements = top_elements
        self.is_top = is_top

        super().__init__(OutKind.CONCEPT_TOP_IN_CATEGORY)

    def severity(self, y: Tensor, rows: Optional[Tensor] = None) -> Tensor:
        """
        Each category uses the Carlini-Wagner metric, and the results of all
        categories are then averaged using the arithmetic mean.
        """
        batch_size = y.shape[0]
        n_categories = len(self.category_indices)

        # `top_elements` [B, Cat]: top element of each seed for each category
        if rows == None:
            top_elements = self.top_elements
            is_top = self.is_top
        else:
            top_elements = self.top_elements[rows]
            is_top = self.is_top[rows]

        # 1. Generate mask for each category
        # Each category contains a 2D tensor - the concept values in that category
        # for each member of the batch
        masks = torch.zeros((n_categories, batch_size, self.num_elements))

        for i, c_indices in enumerate(self.category_indices):
            masks[i, :, c_indices] = y[:, c_indices]

        # 2. Zero out defined top elements in mask for calculation purposes
        masks = masks.scatter_(2, top_elements.T.unsqueeze(dim=2), 0)

        # 3. Calculate argmax for each mask
        # `masks` [C, B]
        masks = masks.max(dim=2)[0]

        # 4. Find spec top concept's value for each category
        # `spec_top_values` [B, C]
        spec_top_values = y.gather(dim=1, index=top_elements)

        # 5. Find difference
        difference = masks.T - spec_top_values

        # 6. Average difference for each element in batch
        num_categories = is_top.sum(dim=1)
        diff_sums = (difference * is_top).sum(dim=1)
        return diff_sums / num_categories
