from act.front_end.concept.output_spec import CombinedOutputSpec
from act.front_end.concept.verifiable_model import ConceptVerifiableModel
from act.front_end.spec_creator_base import LabeledInputTensor
from act.pipeline.fuzzing.actfuzzer import BaseFuzzer, FuzzingConfig
from act.pipeline.fuzzing.concept.property_checker import ConceptPropertyChecker


class ConceptFuzzer(BaseFuzzer):
    def __init__(
        self,
        model: ConceptVerifiableModel,
        initial_seeds: list[LabeledInputTensor],
        config: FuzzingConfig,
    ):
        output_spec = model.output_spec.spec
        assert output_spec is CombinedOutputSpec
        
        property_checker = ConceptPropertyChecker(output_spec)
        
        super().__init__(
            wrapped_model=model,
            initial_seeds=initial_seeds,
            property_checker=property_checker,
            config=config,
        )
