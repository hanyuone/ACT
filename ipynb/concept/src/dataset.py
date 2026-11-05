import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from act.front_end.torchvision_loader.data_model_loader import _resolve_external_dataset
from act.front_end.torchvision_loader.data_model_mapping import get_dataset_info


# From https://github.com/uzukidd/sl-cbm/blob/b0429c55178d98d592fd56bb43e5650236b1aa77/utils/constants.py#L72.
class CUB_features:
    # body part = (min, max)
    has_bill_shape = (0, 8)
    has_wing_color = (9, 23)
    has_upperparts_color = (24, 38)
    has_underparts_color = (39, 53)
    has_breast_pattern = (54, 57)
    has_back_color = (58, 72)
    has_tail_shape = (73, 78)
    has_upper_tail_color = (79, 93)
    has_head_pattern = (94, 104)
    has_breast_color = (105, 119)

    has_throat_color = (120, 134)
    has_eye_color = (135, 148)
    has_bill_length = (149, 151)
    has_forehead_color = (152, 166)
    has_under_tail_color = (167, 181)
    has_nape_color = (182, 196)
    has_belly_color = (197, 211)
    has_wing_shape = (212, 216)
    has_size = (217, 221)
    has_shape = (222, 235)
    has_back_pattern = (236, 239)
    has_tail_pattern = (240, 243)
    has_belly_pattern = (244, 247)
    has_primary_color = (248, 262)
    has_leg_color = (263, 277)
    has_bill_color = (278, 292)
    has_crown_color = (293, 307)
    has_wing_pattern = (308, 311)

    part_mask = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
    93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]  # noqa: RUF012

    # Categories corresponding to body parts (e.g. `has_bill_shape`, `has_bill_length`, `has_bill_color` => `bill`), according to CUB70
    # CUB70 categories are: body, head, neck, beak, tail, right_wing, left_wing, right_leg, left_leg, right_eye, left_eye
    # Left and right body parts are combined
    shape_body = [has_upperparts_color, has_underparts_color, has_breast_pattern, has_back_color, has_breast_color, has_belly_color, has_back_pattern, has_belly_pattern]
    shape_head = [has_head_pattern, has_forehead_color, has_crown_color]
    shape_neck = [has_throat_color, has_nape_color]
    shape_beak = [has_bill_shape, has_bill_length, has_bill_color]
    shape_tail = [has_tail_shape, has_upper_tail_color, has_under_tail_color, has_tail_pattern]
    shape_wing = [has_wing_color, has_wing_shape, has_wing_pattern]
    shape_leg = [has_leg_color]
    shape_eye = [has_eye_color]
    shape_overall = [has_size, has_shape, has_primary_color]

    categories = {
        "body": shape_body,
        "head": shape_head,
        "neck": shape_neck,
        "beak": shape_beak,
        "tail": shape_tail,
        "wings": shape_wing,
        "legs": shape_leg,
        "eyes": shape_eye,
        "overall": shape_overall,
    }

    @staticmethod
    def get_category(concept_index: int) -> str:
        for name, concepts in CUB_features.categories.items():
            for concept_range in concepts:
                cmin, cmax = concept_range

                if cmin <= concept_index <= cmax:
                    return name

        raise ValueError(f"Concept index {concept_index} out of range")


# From https://github.com/mertyg/post-hoc-cbm/blob/c0ba2b7b33b569cc713456cda8c25973809ec7cb/concepts/concept_utils.py#L28.
class EasyDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# From https://github.com/mertyg/post-hoc-cbm/blob/c0ba2b7b33b569cc713456cda8c25973809ec7cb/concepts/concept_utils.py#L34.
class ConceptBank:
    def __init__(self, concept_dict, device):
        all_vectors, concept_names, all_intercepts = [], [], []
        all_margin_info = defaultdict(list)
        for k, (tensor, _, _, intercept, margin_info) in concept_dict.items():
            all_vectors.append(tensor)
            concept_names.append(k)
            all_intercepts.append(np.array(intercept).reshape(1, 1))
            for key, value in margin_info.items():
                if key != "train_margins":
                    all_margin_info[key].append(np.array(value).reshape(1, 1))
        for key, val_list in all_margin_info.items():
            margin_tensor = torch.tensor(
                np.concatenate(val_list, axis=0),
                requires_grad=False,
                dtype=torch.get_default_dtype()
            )
            
            all_margin_info[key] = margin_tensor

        self.concept_info = EasyDict()
        self.concept_info.margin_info = EasyDict(dict(all_margin_info))
        self.concept_info.vectors = torch.tensor(
            np.concatenate(all_vectors, axis=0),
            requires_grad=False,
            dtype=torch.get_default_dtype()
        )
        self.concept_info.norms = torch.norm(
            self.concept_info.vectors, p=2, dim=1, keepdim=True).detach()
        self.concept_info.intercepts = torch.tensor(
            np.concatenate(all_intercepts, axis=0),
            requires_grad=False, dtype=torch.get_default_dtype()
        )
        self.concept_info.concept_names = concept_names
        print("Concept Bank is initialized.")

    def __getattr__(self, item):
        return self.concept_info[item]


class ConceptDataset(Dataset):
    """Given a generic dataset and a """
    def __init__(self, dataset, concepts: list[torch.Tensor]):
        self.dataset = dataset
        self.concepts = concepts

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, _class = self.dataset[index]
        concept = self.concepts[index]

        return image, concept

    def __len__(self):
        return len(self.dataset)


class CubAndMaskDataset(Dataset):
    ACT_DATASET_NAME = "CUB200"

    # Number of testing images in first 70 classes in CUB-200
    FIRST_70_CLASSES_ITEMS = 1_976

    # Indices of images in CUB-70 that have no masks at all
    MISSING_INDICES = {
        *range(243, 273),
        *range(387, 415),
        *range(831, 854),
        566, 586, 599, 1279, 1291, 1297, 1298,
    }

    MASKS = [
        "body", "head", "neck", "beak", "tail", "right_wing", "left_wing",
        "right_leg", "left_leg", "right_eye", "left_eye",
    ]

    @staticmethod
    def _generate_indices() -> list[int]:
        """Return CUB-70 indices with at least one annotation mask."""
        result = list(range(CubAndMaskDataset.FIRST_70_CLASSES_ITEMS))
        return [index for index in result if index not in CubAndMaskDataset.MISSING_INDICES]

    @staticmethod
    def _combine_masks(masks: dict[str, Tensor]) -> dict[str, Tensor]:
        """Combine mask pairs and fill missing masks with the overall mask."""
        overall_mask = torch.clamp(torch.stack(list(masks.values())).sum(dim=0), max=1)

        combined_masks = {
            "body": masks["body"],
            "head": masks["head"],
            "neck": masks["neck"],
            "beak": masks["beak"],
            "tail": masks["tail"],
            "wings": torch.clamp(masks["right_wing"] + masks["left_wing"], max=1),
            "legs": torch.clamp(masks["right_leg"] + masks["left_leg"], max=1),
            "eyes": torch.clamp(masks["right_eye"] + masks["left_eye"], max=1),
        }

        if all((value == 0).all() for value in combined_masks.values()):
            print("Found empty masks set")

        result = {
            name: overall_mask if (mask == 0).all() else mask
            for name, mask in combined_masks.items()
        }
        result["overall"] = overall_mask
        return result

    def _get_masks_for_image(self, image_path: str) -> dict[str, Tensor]:
        print(image_path)
        folder, filename = image_path.split("/")
        folder = int(folder.split(".")[0])
        masks: dict[str, Tensor] = {}

        transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.CenterCrop((224, 224)),
        ])

        for mask_name in CubAndMaskDataset.MASKS:
            mask_path = self.masks_path / f"{folder}/{Path(filename).stem}_{mask_name}.png"
            try:
                with Image.open(mask_path) as mask_image:
                    mask_tensor: Tensor = transform(mask_image).to(device=get_default_device())  # type: ignore
                masks[mask_name] = mask_tensor / 255
            except:
                masks[mask_name] = torch.zeros((3, 224, 224))

        return CubAndMaskDataset._combine_masks(masks)

    def _get_masks(self) -> list[dict[str, Tensor]]:
        """Return masks for selected CUB images in dataset order."""
        split_path = self.cub_path / "CUB_200_2011/train_test_split.txt"
        train_indices = []
        with open(split_path, "r") as split_file:
            for line in split_file:
                index, split = line.split()
                if split == "0":
                    train_indices.append(int(index) - 1)

        train_indices = [train_indices[index] for index in self.indices]
        images_path = self.cub_path / "CUB_200_2011/images.txt"
        images = []
        with open(images_path, "r") as image_file:
            for line in image_file:
                _, image = line.split()
                images.append(image)

        selected_indices = set(train_indices)
        return [
            self._get_masks_for_image(image)
            for index, image in enumerate(images)
            if index in selected_indices
        ]

    def __init__(self, cub_path: Path, masks_path: Path):
        self.indices = CubAndMaskDataset._generate_indices()
        self.cub_path = cub_path
        self.masks_path = masks_path

        dataset_info = get_dataset_info(self.ACT_DATASET_NAME)
        transform = transforms.Compose([
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
        ])

        # In CUB, split value 1 indicates the training dataset.
        cub_train = _resolve_external_dataset(
            dataset_info=dataset_info,
            raw_dir=cub_path,
            transform=transform,
            train=True,
        )
        self.image_dataset = Subset(cub_train, self.indices)
        self.masks_dataset = self._get_masks()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.image_dataset[index], self.masks_dataset[index]
