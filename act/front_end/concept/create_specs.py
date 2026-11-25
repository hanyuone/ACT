import logging
from collections import OrderedDict

import torch
from act.front_end.concept.input_spec import ConceptInputSpec
from act.front_end.concept.output_spec import (
    ConceptOutKind,
    ConceptOutputSpec,
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
        concept_spec_kinds: list[ConceptOutKind],
        epsilons: list[float],
        # Output spec type-specific
        confidence_threshold: float = 0.95,
        ranking_threshold: int = 5,
        categories: list[list[int]] = [],
    ):
        """
        Args:
        - `concept_spec_kinds`: A list of specification types to generate when creating
          specs. Must be all concept-level or all class-level.
        - `epsilons`: List of epsilon values for input-spec mask perturbations.
        - `confidence_threshold` (`CONFIDENCE` only): The confidence value that a concept
          has to have to pass.
        - `ranking_threshold` (`TOP_N` only): The rank that a concept has to be above to pass.
        - `categories` (`TOP_IN_CATEGORY` only): A list of disjoint sets of concepts, with
          each set being a "category" that all categories within belongs to.

        ### Errors
        This function errors if `spec_kinds` are not all concept-level or all class-level.
        """
        super().__init__()

        self.spec_types = concept_spec_kinds

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
    ) -> list[tuple[ConceptOutputSpec, OutputSpec]]:
        """
        Args:
        - `label`: [batch_size]
        """
        concept_output_specs: list[ConceptOutputSpec] = []
        class_output_specs: list[OutputSpec] = []

        for spec_type in self.spec_types:
            match spec_type:
                case ConceptOutKind.CONFIDENCE:
                    threshold = torch.tensor([self.confidence_threshold])
                    concept_output_specs.append(Confidence(num_elements, label, threshold))
                case ConceptOutKind.TOP_N:
                    threshold = torch.tensor([self.ranking_threshold])
                    concept_output_specs.append(TopN(num_elements, label, threshold))
                case ConceptOutKind.TOP_IN_CATEGORY:
                    num_categories = len(self.categories)

                    top_elements = torch.zeros((1, num_categories), dtype=torch.int)
                    is_top = torch.zeros((1, num_categories), dtype=torch.bool)

                    category_index = [
                        label.item() in self.categories[i]
                        for i in range(len(self.categories))
                    ].index(True)

                    top_elements[0, category_index] = label.item()
                    is_top[0, category_index] = True

                    top_in_category = TopInCategory(
                        num_elements, self.categories, top_elements, is_top
                    )
                    # Manually set `top_in_category` for now
                    top_in_category.y_true = label

                    concept_output_specs.append(top_in_category)

        # Currently only supporting TOP1_ROBUST as class spec
        class_output_specs.append(OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=label))

        return [(con, cls) for con in concept_output_specs for cls in class_output_specs]

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

            pairs = [(i, o[0], o[1]) for i in input_specs for o in output_specs]
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

        Args:
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
