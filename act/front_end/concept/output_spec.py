from enum import Enum
from typing import Any, Self, Sequence, cast

from torch import Tensor
import torch

from act.front_end.specs import BaseOutputSpec, OutKind, OutputSpec


class ConceptOutKind(str, Enum):
    CONFIDENCE = "CONFIDENCE"
    TOP_N = "TOP_N"
    TOP_IN_CATEGORY = "TOP_IN_CATEGORY"


class ConceptOutputSpec(BaseOutputSpec[Tensor]):
    def violation(self, y: Tensor, rows: Tensor | None = None) -> tuple[Tensor, Tensor]:
        severity = self.severity(y, rows=rows)
        violated = severity > 0

        return violated, severity

    def encode_linear(self, B: int, n_out: int, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        raise NotImplementedError()

    @classmethod
    def batch(cls, out_specs: Sequence[Self], out_kind: str) -> Self:
        assert out_kind is ConceptOutKind

        match out_kind:
            case ConceptOutKind.CONFIDENCE:
                return Confidence._batch(out_specs)
            case ConceptOutKind.TOP_N:
                return TopN._batch(out_specs)
            case ConceptOutKind.TOP_IN_CATEGORY:
                return TopInCategory._batch(out_specs)


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
        Args:
        - `num_elements`: The number of concepts.
        - `y_true` [B]: The concepts we want to be affected by thresholds.
        - `threshold` [B]: The confidence thresholds of each individual concept.
        """
        if not ((0 <= threshold).all() and (threshold <= 1).all()):
            raise ValueError("Confidence threshold must be in [0, 1]")

        self.num_elements = num_elements
        self.y_true = y_true
        self.threshold = threshold

    def get_kind(self) -> str:
        return "CONFIDENCE"

    def get_y_true(self) -> Tensor | None:
        return self.y_true

    def severity(self, y: Tensor, rows: Tensor | None = None) -> Tensor:
        assert self.y_true is not None

        if rows == None:
            y_true = self.y_true
            threshold = self.threshold
        else:
            y_true = self.y_true[rows]
            threshold = self.threshold[rows]

        batch_size = y.shape[0]
        lb = torch.zeros((batch_size, self.num_elements))
        lb.scatter_(dim=1, index=y_true.unsqueeze(1), src=threshold.unsqueeze(1))

        return (lb - y).max(dim=1, keepdim=False).values

    @classmethod
    def _batch(cls, out_specs: Sequence[Self]) -> Self:
        num_elements = out_specs[0].num_elements
        
        y_true: Tensor = torch.cat([cast(Tensor, o.y_true) for o in out_specs])
        threshold = torch.cat([o.threshold for o in out_specs])
        
        return cls(num_elements, y_true, threshold)


class TopN(ConceptOutputSpec):
    def __init__(self, num_elements: int, y_true: Tensor, threshold: Tensor):
        """
        Args:
        - `num_elements`: The number of concepts.
        - `y_true` [B]: The "correct" concepts we want to be in the top-N.
        - `threshold` [B]: The rank we want our correct concepts to be within.
        """
        if (threshold > num_elements).any():
            raise ValueError("`top_n` must be smaller than number of elements")

        self.num_elements = num_elements
        self.y_true = y_true
        self.threshold = threshold

    def get_kind(self) -> str:
        return "TOP_N"

    def get_y_true(self) -> Tensor | None:
        return self.y_true

    def severity(self, y: Tensor, rows: Tensor | None = None) -> Tensor:
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
            threshold = self.threshold
        else:
            y_true = self.y_true[rows]
            threshold = self.threshold[rows]

        # 1. Get `top_n`-th item from `y`
        sorted = y.sort(dim=1, descending=True)[0].gather(dim=1, index=threshold.unsqueeze(dim=1))

        # 2. Calculate difference
        y_true_confidences = y.gather(dim=1, index=y_true.unsqueeze(dim=1))
        return (sorted - y_true_confidences).squeeze(dim=1)

    @classmethod
    def _batch(cls, out_specs: Sequence[Self]) -> Self:
        num_elements = out_specs[0].num_elements
        assert all(o.num_elements == num_elements for o in out_specs), "Cannot batch top-N specs that have different numbers of elements"
        
        y_true: Tensor = torch.cat([cast(Tensor, o.y_true) for o in out_specs])
        threshold = torch.cat([o.threshold for o in out_specs])
        
        return cls(num_elements, y_true, threshold)


class TopInCategory(ConceptOutputSpec):
    def __init__(
            self,
            num_elements: int,
            category_indices: list[list[int]],
            top_elements: Tensor,
            is_top: Tensor,
        ):
        """
        Args:
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

        # NOTE: strictly speaking, `TOP_IN_CATEGORY` does not have one true value
        self.y_true = None

    def get_kind(self) -> str:
        return "TOP_IN_CATEGORY"

    def get_y_true(self) -> Tensor | None:
        return self.y_true

    def severity(self, y: Tensor, rows: Tensor | None = None) -> Tensor:
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

    @classmethod
    def _batch(cls, out_specs: Sequence[Self]) -> Self:
        num_elements = out_specs[0].num_elements
        assert all(o.num_elements == num_elements for o in out_specs), "Cannot batch top-N specs that have different numbers of elements"

        category_indices = out_specs[0].category_indices
        assert all(o.category_indices == category_indices for o in out_specs), "Cannot batch top-N specs that have different numbers of elements"

        top_elements = torch.cat([o.top_elements for o in out_specs])
        is_top = torch.cat([o.is_top for o in out_specs])

        spec = cls(num_elements, category_indices, top_elements, is_top)

        # TODO: temporary, replace when we think up better way of setting up `y_true`
        if out_specs[0].y_true is not None:
            y_true: Tensor = torch.cat([cast(Tensor, o.y_true) for o in out_specs])
            spec.y_true = y_true

        return spec


class CombinedOutputSpec(BaseOutputSpec[tuple[Tensor, Tensor]]):
    def __init__(self, concept_spec: ConceptOutputSpec, class_spec: OutputSpec):
        self.concept_spec = concept_spec
        self.class_spec = class_spec

    def get_kind(self) -> str:
        return f"COMBINED-{self.concept_spec.get_kind()}-{self.class_spec.get_kind()}"

    def get_y_true(self) -> tuple[Tensor, Tensor] | None:
        concept_y_true = self.concept_spec.get_y_true()
        class_y_true = self.class_spec.get_y_true()

        if concept_y_true is None:
            return None

        if class_y_true is None:
            return None

        return concept_y_true, class_y_true

    def severity(self, y: tuple[Tensor, Tensor], rows: Tensor | None = None) -> Tensor:
        concept_preds, class_preds = y

        concept_severity = self.concept_spec.severity(concept_preds, rows)
        class_severity = self.class_spec.severity(class_preds, rows)

        return torch.max(concept_severity, class_severity)

    def violation(self, y: tuple[Tensor, Tensor], rows: Tensor | None = None) -> tuple[Tensor, Tensor]:
        severity = self.severity(y, rows)
        violated = severity > 0

        return violated, severity

    def encode_linear(self, B: int, n_out: int, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        raise NotImplementedError()

    @staticmethod
    def _get_child_out_kinds(out_kind: str) -> tuple[ConceptOutKind, OutKind] | None:
        _, concept_out_kind, class_out_kind = out_kind.split("-")
        return ConceptOutKind(concept_out_kind), OutKind(class_out_kind)

    @classmethod
    def batch(cls, out_specs: Sequence[Self], out_kind: str) -> Self:
        concept_specs = [o.concept_spec for o in out_specs]
        class_specs = [o.class_spec for o in out_specs]

        out_kinds = cls._get_child_out_kinds(out_kind)
        assert out_kinds is not None
        concept_out_kind, class_out_kind = out_kinds

        return cls(
            ConceptOutputSpec.batch(concept_specs, concept_out_kind),
            OutputSpec.batch(class_specs, class_out_kind)
        )
