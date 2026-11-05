from collections import defaultdict
import os
from pathlib import Path
import pickle
import sys

import torch

act_root = os.path.dirname(os.path.abspath('__file__'))
sys.path.insert(0, act_root) if act_root not in sys.path else None

ACT_ROOT = Path(act_root)

CUB_DATASET_PATH = ACT_ROOT / "data/torchvision/CUB200/raw"
    
def cub_concept_dataset(train: bool) -> list[torch.Tensor]:
    concepts_dict = defaultdict(lambda: torch.zeros(312))

    # Read concepts from file
    attr_labels_path = CUB_DATASET_PATH / "CUB_200_2011/attributes/image_attribute_labels.txt"

    with open(attr_labels_path, "r") as f:
        for line in f.readlines():
            args = line.strip().split()

            if len(args) == 5:
                image_id, attr_id, is_present, certainty, _ = args
            else:
                # `image_attribute_labels.txt` has inconsistencies with images 2275 and 9364
                image_id, attr_id, is_present, certainty, _, _ = args

            if is_present == "1" and int(certainty) >= 3:
                concepts_dict[int(image_id) - 1][int(attr_id) - 1] = 1

    print("Finished reading concepts from file")

    # Filter based on split
    indices = []

    split_path = CUB_DATASET_PATH / "CUB_200_2011/train_test_split.txt"

    with open(split_path, "r") as f:
        for line in f.readlines():
            image_id, is_train = line.strip().split()

            if (train and is_train == "1") or (not train and is_train == "0"):
                indices.append(int(image_id))

    split_concepts = {
        image_id: concepts_dict[image_id]
        for image_id in indices
    }

    print("Finished filtering concepts based on split")

    # Convert back to list of tensors
    sorted_concepts = sorted(split_concepts.items(), key=lambda x: x[0])
    return [tensor for _, tensor in sorted_concepts]


if __name__ == "__main__":
    test_concepts = cub_concept_dataset(train=False)
    pickle.dump(test_concepts, open(ACT_ROOT / "ipynb/slcbm/data/test_312.pkl", "wb"))
    