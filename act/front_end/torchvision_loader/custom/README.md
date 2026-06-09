# Custom dataset types

This folder contains the `Dataset` types for custom datasets, outside of `torchvision.datasets`. To add your own dataset type, please:

1. Make a new folder for your dataset.
2. Create an `__init__.py` inside your folder so that only your dataset is exposed within the folder.
3. Add your dataset to `custom/__init__.py`.

Once that is done, you must add an extra field to your custom dataset definition in `data_model_mapping.py`, `"class_name"`, which is equal to your
dataset class's name. For example, if your dataset class name is `ABCDataset`:

```py
DATASET_MODEL_MAPPING: Dict[str, Dict[str, Any]] = {
    # ...
    "ABC": {
        # ...
        "class_name": "ABCDataset",
    },
    # ...
}
```

Your dataset type must have the following mandatory initialisation arguments (to align with the ones in `torchvision.datasets`), and all other arguments
must be optional:

```py
from torch.util.data import Dataset


class ABCDataset(Dataset):
    def __init__(
        self,
        root: str,
        train: bool,
        download: bool,
        # All other args must be optional:
        foo: str | None = None,
        bar: int = 5,
    ):
        # ...
        pass
```
