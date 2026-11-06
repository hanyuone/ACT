import logging
from collections import OrderedDict

import torch
from act.front_end.concept.input_spec import ConceptInputSpec
from act.front_end.concept.output_spec import (
    Confidence,
    TopInCategory,
    TopN,
)
from act.front_end.spec_creator_base import (
    BaseSpecCreator,
    LabeledInputTensor,
    SpecTuple,
)
from act.front_end.specs import OutKind, OutputSpec
from act.util.device_manager import get_default_device
from torch import Tensor, nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ConceptSpecCreator(BaseSpecCreator):
    """
    Specification creator for concept-based input/output specs.
    """

    # TODO: remove function once YAML integration implemented
    def __init__(
        self,
        spec_kinds: list[OutKind],
        epsilons: list[float],
        # Output spec type-specific
        confidence_threshold: float = 0.95,
        ranking_threshold: int = 5,
        categories: list[list[int]] = [],
    ):
        """
        ### Params
        - `spec_kinds`: A list of specification types to generate when creating
          specs. Must be all concept-level or all class-level.
        - `epsilons`: List of epsilon values for input-spec mask perturbations.

        ### Errors
        This function errors if `spec_kinds` are not all concept-level or all class-level.
        """
        super().__init__()

        assert len(spec_kinds) > 0, "Number of spec kinds cannot be empty"

        is_concept = [st.is_concept() for st in spec_kinds]
        assert all(i == is_concept[0] for i in is_concept), (
            "All spec kinds must be concept- or class-level"
        )

        self.is_concept_level = is_concept[0]
        self.spec_types = spec_kinds

        self.epsilons = epsilons

        self.confidence_threshold = confidence_threshold
        self.ranking_threshold = ranking_threshold
        self.categories = categories

    def _create_input_specs(
        self, sample: Tensor, mask: Tensor
    ) -> list[ConceptInputSpec]:
        input_specs: list[ConceptInputSpec] = []

        for eps in self.epsilons:
            input_specs.append(ConceptInputSpec(sample, mask, eps))

        return input_specs

    def _create_output_specs(
        self, num_elements: int, label: Tensor
    ) -> list[OutputSpec]:
        """
        ### Params
        - `label`: [batch_size]
        """
        output_specs: list[OutputSpec] = []

        if self.is_concept_level:
            for spec_type in self.spec_types:
                match spec_type:
                    case OutKind.CONCEPT_CONFIDENCE:
                        threshold = torch.tensor([self.confidence_threshold])
                        output_specs.append(Confidence(num_elements, label, threshold))
                    case OutKind.CONCEPT_TOP_N:
                        threshold = torch.tensor([self.ranking_threshold])
                        output_specs.append(TopN(num_elements, label, threshold))
                    case OutKind.CONCEPT_TOP_IN_CATEGORY:
                        num_categories = len(self.categories)

                        top_elements = torch.zeros((1, num_categories), dtype=torch.int)
                        is_top = torch.zeros((1, num_categories), dtype=torch.bool)

                        category_index = [
                            label.item() in self.categories[i]
                            for i in range(len(self.categories))
                        ].index(True)

                        top_elements[0, category_index] = label.item()
                        is_top[0, category_index] = True

                        output_specs.append(
                            TopInCategory(
                                num_elements, self.categories, top_elements, is_top
                            )
                        )
        else:
            for spec_type in self.spec_types:
                match spec_type:
                    case OutKind.TOP1_ROBUST:
                        output_specs.append(OutputSpec(kind=spec_type, y_true=label))

        return output_specs

    def _dataset_model_spec(
        self,
        dl_key: str,
        model_key: str,
        dataloader: DataLoader,
        model: nn.Module,
        num_elements: int,
        num_samples: int,
        start_index: int,
    ) -> SpecTuple:
        labeled_tensors = []
        spec_pairs = []

        for idx, (images, masks, targets) in enumerate(dataloader):
            if idx < start_index:
                continue
            if len(labeled_tensors) >= num_samples:
                break

            # DataLoader always returns CPU tensors regardless of torch.set_default_device
            device = get_default_device()
            images = images.to(device)
            targets = targets.to(device)

            labeled_tensors.append(LabeledInputTensor(tensor=images, label=targets))

            input_specs = self._create_input_specs(images, masks)
            output_specs = self._create_output_specs(num_elements, targets)

            pairs = self._create_spec_combinations(input_specs, output_specs)
            spec_pairs.extend(pairs)

        if not spec_pairs:
            raise RuntimeError(f"No valid specs generated for {dl_key} + {model_key}")

        logger.info(f"Generated {len(spec_pairs)} spec combinations")

        return dl_key, model_key, model, labeled_tensors, spec_pairs

    def create_specs_for_data_model_pairs(
        self,
        dataloaders: OrderedDict[str, tuple[DataLoader, int]],
        models: OrderedDict[str, nn.Module],
        num_samples: int = 10,
        start_index: int = 0,
    ) -> list[SpecTuple]:
        """
        Create sets of specs for dataset/model pairs. Creates `SpecTuple`s for
        concept-level or class-level output specs, which can be specified in the
        `spec_types` provided.

        # Arguments
        - `datasets`: A list of datasets with output `(image, mask, labels)`. If generating
          concept-level specs, provide *concept* ground-truth labels, otherwise provide
          *class* ground-truth labels.
        - `models`: A list of PyTorch models.
        - `num_samples`: The number of samples to take for each dataset/model pair.
        - `start_index`: The starting index at which we take our samples.
        """
        results: list[SpecTuple] = []

        for dl_key, (dataloader, num_elements) in dataloaders.items():
            for model_key, model in models.items():
                print(f"Generating specs for {dl_key}, {model_key}")
                result = self._dataset_model_spec(
                    dl_key,
                    model_key,
                    dataloader,
                    model,
                    num_elements,
                    num_samples,
                    start_index,
                )
                results.append(result)

        return results
