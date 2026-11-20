from typing import Any

from torch import Tensor, nn
from act.front_end.verifiable_model import InputLayer, InputSpecLayer, OutputSpecLayer, VerifiableModel


class ConceptVerifiableModel(VerifiableModel):
    def __init__(
        self,
        input_layer: InputLayer,
        input_spec: InputSpecLayer,
        model: nn.Module,
        concept_output_spec: OutputSpecLayer,
        class_output_spec: OutputSpecLayer
    ):
        """
        ### Arguments
        - `model`: concept-based model, returns (class_preds, concept_preds)
        """
        self.input_layer = input_layer
        self.input_spec = input_spec
        self.model = model
        self.concept_output_spec = concept_output_spec
        self.class_output_spec = class_output_spec

    def forward(self, x, rows: Tensor | None = None) -> dict[str, Any]:
        x = self.input_layer(x)
        
        input_result = (
            self.input_spec(x)
            if rows is None
            else self.input_spec(x, rows=rows)
        )
        x, input_satisfied, input_explanation = input_result
    
        x_concept, x_class = self.model(x)

        concept_output_result = self.concept_output_spec(x_concept)
        x_class, concept_output_satisfied, concept_output_explanation = concept_output_result
        concept_output_satisfied = concept_output_satisfied.detach().bool()

        class_output_result = self.class_output_spec(x_class)
        x_class, class_output_satisfied, class_output_explanation = class_output_result
        class_output_satisfied = class_output_satisfied.detach().bool()

        return {
            "output": (x_concept, x_class),
            "input_satisfied": input_satisfied,
            "input_explanation": input_explanation,
            "concept_output_satisfied": concept_output_satisfied,
            "concept_output_explanation": concept_output_explanation,
            "class_output_satisfied": class_output_satisfied,
            "class_output_explanation": class_output_explanation,
        }
