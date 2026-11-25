from dataclasses import dataclass
from pathlib import Path
from typing import Self, Sequence

import torch

from act.front_end.concept.output_spec import CombinedOutputSpec
from act.pipeline.fuzzing.checker import BaseCE, BasePropertyChecker
from act.pipeline.fuzzing.corpus import FuzzingSeed


@dataclass
class ConceptCE(BaseCE):
    input: torch.Tensor
    output: torch.Tensor

    seed_input: torch.Tensor
    kind: str
    spec_row: int
    severity: float
    timestamp: float

    true_concept: int | None = None
    true_class: int | None = None

    def summary(self) -> str:
        return super().summary()
    
    def to_dict(self) -> dict[str, object]:
        return super().to_dict()
    
    @classmethod
    def load(cls, path: str | Path) -> Self:
        return super().load(path)


class ConceptPropertyChecker(BasePropertyChecker[tuple[torch.Tensor, torch.Tensor]]):
    """
    For `(concept_preds, class_preds)` and an `output_spec`, checks 
    """
    def __init__(self, spec: CombinedOutputSpec):
        self.spec = spec

    def check(
        self,
        inputs: torch.Tensor,
        outputs: tuple[torch.Tensor, torch.Tensor],
        seeds: FuzzingSeed
    ) -> tuple[torch.Tensor, Sequence[BaseCE]]:
        if self.spec is None:
            raise ValueError("PropertyChecker requires an OutputSpec")

        input_satisfied = getattr(outputs, INPUT_FEASIBILITY_ATTR, None)
        if not isinstance(input_satisfied, torch.Tensor):
            raise RuntimeError(
                "PropertyChecker requires lane-aware input feasibility metadata "
                "from VerifiableModel.forward"
            )
        input_satisfied = input_satisfied.to(
            device=outputs.device, dtype=torch.bool
        ).reshape(-1)
        if input_satisfied.shape != (inputs.shape[0],):
            raise RuntimeError(
                "Input feasibility mask must have shape "
                f"({inputs.shape[0]},), got {tuple(input_satisfied.shape)}"
            )
        
        violations_mask, severity = self.spec.violation(
            outputs, rows=seeds.original_index
        )
        rejected_mask = violations_mask & ~input_satisfied
        rejected_count = int(rejected_mask.sum().item())
        if rejected_count:
            self.infeasible_candidates += rejected_count
            print(
                "⚠️  [PropertyChecker] Rejected "
                f"{rejected_count} infeasible counterexample candidate(s) "
                f"(total: {self.infeasible_candidates})"
            )
        violations_mask = violations_mask & input_satisfied
        return self._build_results(
            inputs, outputs, violations_mask, severity, seeds=seeds
        )
