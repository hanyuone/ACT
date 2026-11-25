from act.front_end.concept.output_spec import CombinedOutputSpec
from act.front_end.verifiable_model import VerifiableModel


class ConceptVerifiableModel(VerifiableModel[CombinedOutputSpec]):
    """
    A wrapper for `VerifiableModel` that contains type annotations for each layer.
    """
    pass
