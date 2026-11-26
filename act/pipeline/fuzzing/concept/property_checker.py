import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import torch

from act.front_end.concept.output_spec import CombinedOutputSpec
from act.pipeline.fuzzing.checker import BaseCE, BasePropertyChecker
from act.pipeline.fuzzing.corpus import FuzzingSeed


@dataclass
class ConceptCE(BaseCE):
    input: torch.Tensor
    output: tuple[torch.Tensor, torch.Tensor]

    seed_input: torch.Tensor
    kind: str
    spec_row: int
    severity: float
    timestamp: float

    true_concept: int | None = None
    true_class: int | None = None

    def summary(self) -> str:
        details = f"spec_row={self.spec_row}, severity={self.severity:.6g}"

        if self.true_concept is None or self.true_class is None:
            return f"{self.kind}: {details}"

        concept_probs, class_probs = self.output
        concept_pred = concept_probs.argmax().item()
        class_pred = class_probs.argmax().item()

        return (
            f"{self.kind}: predicted ({concept_pred}, {class_pred}),"
            f"expected ({self.true_concept}, {self.true_class}) ({details})"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "input": self.input,
            "output": self.output,
            "seed_input": self.seed_input,
            "kind": self.kind,
            "spec_row": self.spec_row,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "true_concept": self.true_concept,
            "true_class": self.true_class,
        }

    @classmethod
    def load(cls, path: str | Path) -> Self:
        data = cast(dict[str, Any], torch.load(path))

        return cls(
            input=data["input"],
            output=data["output"],
            seed_input=data["seed_input"],
            kind=data["kind"],
            spec_row=data["spec_row"],
            severity=data["severity"],
            timestamp=data["timestamp"],
            true_concept=data["true_concept"],
            true_class=data["true_class"],
        )


class ConceptPropertyChecker(BasePropertyChecker[tuple[torch.Tensor, torch.Tensor]]):
    """
    For `(concept_preds, class_preds)` and an `output_spec`, checks
    """

    def __init__(self, spec: CombinedOutputSpec):
        self.spec = spec
        self.infeasible_candidates = 0

    def check(
        self,
        inputs: torch.Tensor,
        outputs: tuple[torch.Tensor, torch.Tensor],
        seeds: FuzzingSeed,
        input_satisfied: torch.Tensor,
    ) -> tuple[torch.Tensor, Sequence[BaseCE]]:
        if self.spec is None:
            raise ValueError("PropertyChecker requires an OutputSpec")

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

    def _build_results(
        self,
        inputs: torch.Tensor,
        outputs: tuple[torch.Tensor, torch.Tensor],
        violations_mask: torch.Tensor,
        severity: torch.Tensor,
        seeds: FuzzingSeed,
    ) -> tuple[torch.Tensor, list[ConceptCE]]:
        """Build Counterexample list from violation mask and return (mask, list)."""
        timestamp = time.time()
        violation_indices = violations_mask.nonzero(as_tuple=True)[0]

        y_true = self.spec.get_y_true()
        true_classes, true_concepts = (None, None) if y_true is None else y_true

        counterexamples: list[ConceptCE] = []

        for idx in violation_indices:
            i = int(idx.item())
            spec_row = int(seeds.original_index[i].item())

            if true_classes is None:
                true_class = None
            else:
                true_class = int(true_classes[spec_row].item())

            if true_concepts is None:
                true_concept = None
            else:
                true_concept = int(true_concepts[spec_row].item())

            counterexamples.append(
                ConceptCE(
                    input=inputs[i].detach().cpu(),
                    output=(outputs[0][i].detach().cpu(), outputs[1][i].detach().cpu()),
                    seed_input=seeds.original_tensor[i].detach().cpu(),
                    kind=self.spec.get_kind(),
                    spec_row=spec_row,
                    severity=float(severity[i].item()),
                    timestamp=timestamp,
                    true_class=true_class,
                    true_concept=true_concept,
                )
            )

        return violations_mask, counterexamples
