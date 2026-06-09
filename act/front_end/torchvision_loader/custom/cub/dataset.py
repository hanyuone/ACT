"""
Code modified from https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/dataset.py
General utils for training, evaluation and data loading
"""
import os
import pickle
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, extract_archive

from .data_processing import extract_data


N_ATTRIBUTES = 312

class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    RAW_DATASET_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    MD5 = "97eceeb196236b17998738112f37df78"

    # Modified to be in format
    # ```
    # dataset_class(
    #     root=str(raw_dir),
    #     train=True,
    #     download=True
    # )
    # ```
    def __init__(
            self,
            root,
            train,
            download,
            # Use sensible defaults
            use_attr=False,
            no_img=False,
            uncertain_label=False,
            image_dir="images",
            n_class_attr=1,
            transform=None,
        ):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.root = root
        self.train = train

        if download:
            self.download_and_process()

        if self.train:
            self.data = pickle.load(open(f"{root}/processed/train", "rb"))
        else:
            self.data = pickle.load(open(f"{root}/processed/test", "rb"))

        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr

    def download_and_process(self):
        if check_integrity(f"{self.root}/CUB_200_2011.tgz", self.MD5):
            return
        
        download_and_extract_archive(
            url=self.RAW_DATASET_URL,
            download_root=self.root,
            extract_root=f"{self.root}/decompressed"
        )

        train, _, test = extract_data(f"{self.root}/decompressed")
        
        os.mkdir(f"{self.root}/processed")
        pickle.dump(train, open(f"{self.root}/processed/train", "wb"))
        pickle.dump(test, open(f"{self.root}/processed/test", "wb"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        # Trim unnecessary paths
        try:
            idx = img_path.split('/').index('CUB_200_2011')
            if self.image_dir != 'images':
                img_path = '/'.join([self.image_dir] + img_path.split('/')[idx+1:])
                img_path = img_path.replace('images/', '')
            else:
                img_path = '/'.join(img_path.split('/')[idx:])
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'train' if self.train else 'test'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        if self.transform:
            img = self.transform(img)

        if self.use_attr:
            if self.uncertain_label:
                attr_label = img_data['uncertain_attribute_label']
            else:
                attr_label = img_data['attribute_label']
            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros((N_ATTRIBUTES, self.n_class_attr))
                    one_hot_attr_label[np.arange(N_ATTRIBUTES), attr_label] = 1
                    return one_hot_attr_label, class_label
                else:
                    return attr_label, class_label
            else:
                return img, class_label, attr_label
        else:
            return img, class_label
