import os
from pathlib import Path
import pickle
from pytorchcv.model_provider import get_model
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from act.front_end.torchvision_loader.data_model_loader import _resolve_external_dataset
from act.front_end.torchvision_loader.data_model_mapping import DATASET_MODEL_MAPPING
from act.util.device_manager import initialize_device
from ipynb.slcbm.src.dataset import ConceptBank
from ipynb.slcbm.src.model import SLCBM, ResNetBottom


# Set path for ACT root
act_root = os.path.dirname(os.path.abspath('__file__'))
sys.path.insert(0, act_root) if act_root not in sys.path else None

ACT_ROOT = Path(act_root)

# Classes
class ConceptDataset(Dataset):
    def __init__(self, base: Dataset, concepts):
        self.base = base
        self.concepts = concepts

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, label = self.base[index]
        concept = self.concepts[index]

        return (image, label, concept)

    def __len__(self) -> int:
        return len(self.concepts)


class PELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, input: torch.Tensor, softmax_space: bool = True):
        """
            input: [B, C, H, W]
        """
        B, C, H, W = input.shape

        input = input.reshape(B, C, H * W)

        if not softmax_space:
            input = F.softmax(input, dim=2)
        
        logs = torch.log10(input + 1e-10)
        concept_entropy = -torch.sum(input * logs, dim=2)
        entropy = concept_entropy.mean(dim=1)
        
        return entropy


class CBMLoss(nn.Module):
    """
    Loss function for SL-CBM as defined in (Zhang et al. 2026).
    """
    def __init__(self):
        super().__init__()

        # Class accuracy: cross-entropy loss
        self.ce_param = 1.0
        self.ce_loss = nn.CrossEntropyLoss()

        # Concept accuracy
        self.ca_param = 1.0
        self.ca_loss = nn.L1Loss()

        # Saliency sparsity: entropy loss
        self.e_param = 1.0
        self.e_loss = PELoss()

    def forward(
        self,
        output: tuple[torch.Tensor, torch.Tensor],
        ground: tuple[torch.Tensor, torch.Tensor],
        saliency_maps: torch.Tensor
    ):
        class_loss = self.ce_param * self.ce_loss(output[0], ground[0])
        concept_loss = self.ca_param * self.ca_loss(output[1], ground[1])
        saliency_sparsity = (self.e_param * self.e_loss(saliency_maps, False)).mean()

        return class_loss + concept_loss + saliency_sparsity

# Set default device and float type
DEVICE = "mps"
DTYPE = "float32"

# Initialise bottleneck model
BACKBONE_ROOT = ACT_ROOT / "data/torchvision/CUB200/models"
CONCEPT_BANKS_PATH = ACT_ROOT / "ipynb/slcbm/data/cub_resnet18_cub_0.01_100.pkl"
CUB_WEIGHTS_PATH = ACT_ROOT / "data/data_ckpt/CUB/spss_resnet_cbm_TCAV_1.0_1.0_1.0/trainable_weights.pt"

BATCH_SIZE = 1
DATASET_NAME = "CUB200"
CUB_MEAN = [0.5, 0.5, 0.5]
CUB_STD = [2.0, 2.0, 2.0]

NUM_CONCEPTS = 312
NUM_EPOCHS = 1000
BATCH_SIZE = 32

# Set up default device and dtype
initialize_device(device=DEVICE, dtype=DTYPE)

# Set up testing CUB dataset, using ACT-internal functions
dataset_info = DATASET_MODEL_MAPPING[DATASET_NAME]

transform = transforms.Compose([
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor()
])

base_dataset = _resolve_external_dataset(
    dataset_info=dataset_info,
    raw_dir = ACT_ROOT / "data/torchvision/CUB200/raw",
    transform=transform,
    train=True,
)

concepts: list[torch.Tensor] = pickle.load(open(ACT_ROOT / "ipynb/slcbm/data/train_312.pkl", "rb"))

dataset = ConceptDataset(base_dataset, concepts)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator(device='mps:0')
)

# Set up concept bottleneck model
backbone_model = get_model(
    name="resnet18_cub",
    pretrained=True,
    root=BACKBONE_ROOT
)

backbone = ResNetBottom(backbone_model)
backbone.to(device=DEVICE).eval()
backbone.encode_image = lambda image: backbone(image) # type: ignore

normalizer = transforms.Normalize(mean=CUB_MEAN, std=CUB_STD)

with open(CONCEPT_BANKS_PATH, "rb") as f:
    concept_dict = pickle.load(f)

concept_bank = ConceptBank(
    concept_dict=concept_dict,
    device=torch.float32
)

bottleneck_model = SLCBM(
    backbone=backbone,
    normalizer=normalizer,
    concept_bank=concept_bank,
    num_classes=200
)
bottleneck_model.to(device=DEVICE)

# Model tuning
loss_fn = CBMLoss()
optimiser = torch.optim.SGD(bottleneck_model.parameters())

for epoch_idx in range(NUM_EPOCHS):
    print(f'EPOCH {epoch_idx + 1}:')

    running_loss = 0.
    batch_loss = 0.

    # Train one epoch
    for loader_idx, data in enumerate(dataloader):
        optimiser.zero_grad()

        # images: [B, C, H, W]
        # ground_labels: [B]
        images, ground_labels, ground_concepts = data
        images: torch.Tensor = images

        # class_preds: [B, X]
        # concept_preds: [B, Y]
        # saliency_maps: [B, H, W, O]
        class_preds, concept_preds, saliency_maps = bottleneck_model(images.to(device=DEVICE))
        output = (class_preds, concept_preds)

        saliency_maps: torch.Tensor = saliency_maps
        # saliency_maps: [B, O, H, W]
        saliency_maps = saliency_maps.permute(0, 3, 1, 2)

        # ground_concepts: [B, Y]
        ground = (ground_labels.to(device=DEVICE), ground_concepts.to(device=DEVICE))

        # loss: [B]
        loss = loss_fn(output, ground, saliency_maps)
        loss.backward()

        optimiser.step()

        # Gather data and report
        running_loss += loss.item()
        if loader_idx % 50 == 49:
            batch_loss = running_loss / 50 # loss per batch
            print(f'  batch {loader_idx + 1} loss: {batch_loss}')
            running_loss = 0.

    # Total loss
    total_loss = 0.

    with torch.no_grad():
        for i, vdata in enumerate(dataloader):
            # images: [B, C, H, W]
            # ground_labels: [B]
            images, ground_labels, ground_concepts = vdata
            images: torch.Tensor = images

            # class_preds: [B, X]
            # concept_preds: [B, Y]
            # saliency_maps: [B, H, W, O]
            class_preds, concept_preds, saliency_maps = bottleneck_model(images.to(device=DEVICE))
            voutput = (class_preds, concept_preds)

            saliency_maps: torch.Tensor = saliency_maps
            # saliency_maps: [B, O, H, W]
            saliency_maps = saliency_maps.permute(0, 3, 1, 2)

            # ground_concepts: [B, Y]
            vground = (ground_labels.to(device=DEVICE), ground_concepts.to(device=DEVICE))

            vloss = loss_fn(voutput, vground, saliency_maps)
            total_loss += vloss

    average_loss = total_loss / (len(dataloader) + 1)
    print(f"Epoch loss: {average_loss}")

torch.save(bottleneck_model.state_dict(), ACT_ROOT / "ipynb/slcbm/data/tuned_weights.pt")
