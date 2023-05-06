
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms

from .iqa_dataset import *
from .iqa_dataset_examplar import *
from .iqa_dataset_a_dif import *
from .iqa_dataset_b_dif import *
from .iqa_dataset_dif import *
from .iqa_dataset_dif_partial import *
from .iqa_dataset_dif_student import *
from .iqa_dataset_dif_teacher import *
from .samplers import SubsetRandomSampler, IQAPatchDistributedSampler

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == "bicubic":
            return InterpolationMode.BICUBIC
        elif method == "lanczos":
            return InterpolationMode.LANCZOS
        elif method == "hamming":
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR

    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_transform(dataset, is_train, config):
    if dataset == "koniq":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif dataset == "livec":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif dataset == "live":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif dataset == "tid2013":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif dataset == "csiq":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif dataset == "kadid":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    if dataset == "spaq":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    # transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    # transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    if dataset == "livefb":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize((512, 512)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((512, 512)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    return transform


def build_transform_teacher(dataset, is_train, config):
    if dataset == "koniq":
        if is_train:
            transform = transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(),
                    transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif dataset == "livec":
        if is_train:
            transform = transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif dataset == "live":
        if is_train:
            transform = transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif dataset == "tid2013":
        if is_train:
            transform = transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif dataset == "csiq":
        if is_train:
            transform = transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif dataset == "kadid":
        if is_train:
            transform = transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    if dataset == "spaq":
        if is_train:
            transform = transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(),
                    # transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    # transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    if dataset == "livefb":
        if is_train:
            transform = transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(),
                    transforms.Resize((512, 512)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((512, 512)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    return transform


def build_transform_student(dataset, is_train, config):
    if dataset == "koniq":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomRotation((45, 90)),
                    transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif dataset == "livec":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomRotation((45, 90)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif dataset == "live":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomRotation((45, 90)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif dataset == "tid2013":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomRotation((45, 90)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif dataset == "csiq":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomRotation((45, 90)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif dataset == "kadid":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomRotation((45, 90)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    if dataset == "spaq":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomRotation((45, 90)),
                    # transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    # transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    if dataset == "livefb":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomRotation((45, 90)),
                    transforms.Resize((512, 512)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((512, 512)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    return transform


# def IQA_build_loader_knn(config, pseudo_label_data, examplar_total=None):
#     print("pseudo_label_data", type(pseudo_label_data))
#     print("examplar_total", type(examplar_total))
#     if examplar_total:
#         dataset_train = pseudo_label_data + examplar_total
#     else:
#         dataset_train = pseudo_label_data
def IQA_build_loader_knn(config, dataset_train):

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == "part":
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE1,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    # setup mixup / cutmix
    mixup_fn = None

    return data_loader_train, mixup_fn


def build_IQA_dataset_krr_student(config, train_index, pseudo_label, examplar=None):
    print(config.DATA.DATASET1)
    if config.DATA.DATASET1 == "koniq":
        train_dataset = KONIQDATASET7(
            config.DATA.DATA_PATH1,
            train_index,
            config.DATA.PATCH_NUM1,
            examplar,
            transform=build_transform(config.DATA.DATASET1, is_train=True, config=config),
            pseudo_label=pseudo_label
        )
    elif config.DATA.DATASET1 == "livec":
        train_dataset = LIVECDATASET7(
            config.DATA.DATA_PATH1,
            train_index,
            config.DATA.PATCH_NUM1,
            examplar,
            transform=build_transform(config.DATA.DATASET1, is_train=True, config=config),
            pseudo_label=pseudo_label
        )
    elif config.DATA.DATASET1 == "live":
        train_dataset = LIVEDataset7(
            config.DATA.DATA_PATH1,
            train_index,
            config.DATA.PATCH_NUM1,
            examplar,
            transform=build_transform(config.DATA.DATASET1, is_train=True, config=config),
            pseudo_label=pseudo_label
        )
    elif config.DATA.DATASET1 == "tid2013":
        train_dataset = TID2013Dataset7(
            config.DATA.DATA_PATH1,
            train_index,
            config.DATA.PATCH_NUM1,
            examplar,
            transform=build_transform(config.DATA.DATASET1, is_train=True, config=config),
            pseudo_label=pseudo_label
        )
    elif config.DATA.DATASET1 == "csiq":
        train_dataset = CSIQDataset7(
            config.DATA.DATA_PATH1,
            train_index,
            config.DATA.PATCH_NUM1,
            examplar,
            transform=build_transform(config.DATA.DATASET1, is_train=True, config=config),
            pseudo_label=pseudo_label
        )
    elif config.DATA.DATASET1 == "kadid":
        train_dataset = KADIDDataset7(
            config.DATA.DATA_PATH1,
            train_index,
            config.DATA.PATCH_NUM1,
            examplar,
            transform=build_transform(config.DATA.DATASET1, is_train=True, config=config),
            pseudo_label=pseudo_label
        )
    elif config.DATA.DATASET1 == "spaq":
        train_dataset = SPAQDATASET7(
            config.DATA.DATA_PATH1,
            train_index,
            config.DATA.PATCH_NUM1,
            examplar,
            transform=build_transform(config.DATA.DATASET1, is_train=True, config=config),
            pseudo_label=pseudo_label
        )
    elif config.DATA.DATASET1 == "livefb":
        train_dataset = FBLIVEFolder7(
            config.DATA.DATA_PATH1,
            train_index,
            config.DATA.PATCH_NUM1,
            examplar,
            transform=build_transform(config.DATA.DATASET1, is_train=True, config=config),
            pseudo_label=pseudo_label
        )
    else:
        raise NotImplementedError("We only support common IQA dataset Now.")
    return train_dataset


def build_IQA_dataset_krr_teacher(config, train_index):
    print(config.DATA.DATASET1)
    if config.DATA.DATASET1 == "koniq":
        train_dataset = KONIQDATASET8(
            config.DATA.DATA_PATH1,
            train_index,
            config.DATA.PATCH_NUM1,
            transform=build_transform_teacher(config.DATA.DATASET1, is_train=True, config=config)
        )
    elif config.DATA.DATASET1 == "livec":
        train_dataset = LIVECDATASET8(
            config.DATA.DATA_PATH1,
            train_index,
            config.DATA.PATCH_NUM1,
            transform=build_transform_teacher(config.DATA.DATASET1, is_train=True, config=config)
        )
    elif config.DATA.DATASET1 == "live":
        train_dataset = LIVEDataset8(
            config.DATA.DATA_PATH1,
            train_index,
            config.DATA.PATCH_NUM1,
            transform=build_transform_teacher(config.DATA.DATASET1, is_train=True, config=config)
        )
    elif config.DATA.DATASET1 == "tid2013":
        train_dataset = TID2013Dataset8(
            config.DATA.DATA_PATH1,
            train_index,
            config.DATA.PATCH_NUM1,
            transform=build_transform_teacher(config.DATA.DATASET1, is_train=True, config=config)
        )
    elif config.DATA.DATASET1 == "csiq":
        train_dataset = CSIQDataset8(
            config.DATA.DATA_PATH1,
            train_index,
            config.DATA.PATCH_NUM1,
            transform=build_transform_teacher(config.DATA.DATASET1, is_train=True, config=config)
        )
    elif config.DATA.DATASET1 == "kadid":
        train_dataset = KADIDDataset8(
            config.DATA.DATA_PATH1,
            train_index,
            config.DATA.PATCH_NUM1,
            transform=build_transform_teacher(config.DATA.DATASET1, is_train=True, config=config)
        )
    elif config.DATA.DATASET1 == "spaq":
        train_dataset = SPAQDATASET8(
            config.DATA.DATA_PATH1,
            train_index,
            config.DATA.PATCH_NUM1,
            transform=build_transform_teacher(config.DATA.DATASET1, is_train=True, config=config)
        )
    elif config.DATA.DATASET1 == "livefb":
        train_dataset = FBLIVEFolder8(
            config.DATA.DATA_PATH1,
            train_index,
            config.DATA.PATCH_NUM1,
            transform=build_transform_teacher(config.DATA.DATASET1, is_train=True, config=config)
        )
    else:
        raise NotImplementedError("We only support common IQA dataset Now.")
    return train_dataset


def build_IQA_dataset_a_krr(config):
    print(config.DATA.DATASET)
    if config.DATA.DATASET == "koniq":
        train_dataset = KONIQDATASET5(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(config.DATA.DATASET, is_train=True, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET, is_train=True, config=config),
        )
        test_dataset = KONIQDATASET5(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(config.DATA.DATASET, is_train=False, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET, is_train=False, config=config),
        )
    elif config.DATA.DATASET == "livec":
        train_dataset = LIVECDATASET5(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(config.DATA.DATASET, is_train=True, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET, is_train=True, config=config),
        )
        test_dataset = LIVECDATASET5(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(config.DATA.DATASET, is_train=False, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET, is_train=False, config=config),
        )
    elif config.DATA.DATASET == "live":
        train_dataset = LIVEDataset5(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(config.DATA.DATASET, is_train=True, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET, is_train=True, config=config),
        )
        test_dataset = LIVEDataset5(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(config.DATA.DATASET, is_train=False, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET, is_train=False, config=config),
        )
    elif config.DATA.DATASET == "tid2013":
        train_dataset = TID2013Dataset5(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(config.DATA.DATASET, is_train=True, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET, is_train=True, config=config),
        )
        test_dataset = TID2013Dataset5(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(config.DATA.DATASET, is_train=False, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET, is_train=False, config=config),
        )
    elif config.DATA.DATASET == "csiq":
        train_dataset = CSIQDataset5(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(config.DATA.DATASET, is_train=True, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET, is_train=True, config=config),
        )
        test_dataset = CSIQDataset5(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(config.DATA.DATASET, is_train=False, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET, is_train=False, config=config),
        )
    elif config.DATA.DATASET == "kadid":
        train_dataset = KADIDDataset5(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(config.DATA.DATASET, is_train=True, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET, is_train=True, config=config),
        )
        test_dataset = KADIDDataset5(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(config.DATA.DATASET, is_train=False, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET, is_train=False, config=config),
        )
    elif config.DATA.DATASET == "spaq":
        train_dataset = SPAQDATASET5(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(config.DATA.DATASET, is_train=True, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET, is_train=True, config=config),
        )
        test_dataset = SPAQDATASET5(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(config.DATA.DATASET, is_train=False, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET, is_train=False, config=config),
        )
    elif config.DATA.DATASET == "livefb":
        train_dataset = FBLIVEFolder5(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(config.DATA.DATASET, is_train=True, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET, is_train=True, config=config),
        )
        test_dataset = FBLIVEFolder5(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(config.DATA.DATASET, is_train=False, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET, is_train=False, config=config),
        )
    else:
        raise NotImplementedError("We only support common IQA dataset Now.")

    return train_dataset, test_dataset


def build_IQA_dataset_b_krr(config):
    print(config.DATA.DATASET1)
    if config.DATA.DATASET1 == "koniq":
        train_dataset = KONIQDATASET6(
            config.DATA.DATA_PATH1,
            config.SET1.TRAIN_INDEX,
            config.DATA.PATCH_NUM1,
            transform=build_transform(config.DATA.DATASET1, is_train=True, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET1, is_train=True, config=config),
        )
        test_dataset = KONIQDATASET6(
            config.DATA.DATA_PATH1,
            config.SET1.TEST_INDEX,
            config.DATA.PATCH_NUM1,
            transform=build_transform(config.DATA.DATASET1, is_train=False, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET1, is_train=False, config=config),
        )
    elif config.DATA.DATASET1 == "livec":
        train_dataset = LIVECDATASET6(
            config.DATA.DATA_PATH1,
            config.SET1.TRAIN_INDEX,
            config.DATA.PATCH_NUM1,
            transform=build_transform(config.DATA.DATASET1, is_train=True, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET1, is_train=True, config=config),
        )
        test_dataset = LIVECDATASET6(
            config.DATA.DATA_PATH1,
            config.SET1.TEST_INDEX,
            config.DATA.PATCH_NUM1,
            transform=build_transform(config.DATA.DATASET1, is_train=False, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET1, is_train=False, config=config),
        )
    elif config.DATA.DATASET1 == "live":
        train_dataset = LIVEDataset6(
            config.DATA.DATA_PATH1,
            config.SET1.TRAIN_INDEX,
            config.DATA.PATCH_NUM1,
            transform=build_transform(config.DATA.DATASET1, is_train=True, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET1, is_train=True, config=config),
        )
        test_dataset = LIVEDataset6(
            config.DATA.DATA_PATH1,
            config.SET1.TEST_INDEX,
            config.DATA.PATCH_NUM1,
            transform=build_transform(config.DATA.DATASET1, is_train=False, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET1, is_train=False, config=config),
        )
    elif config.DATA.DATASET1 == "tid2013":
        train_dataset = TID2013Dataset6(
            config.DATA.DATA_PATH1,
            config.SET1.TRAIN_INDEX,
            config.DATA.PATCH_NUM1,
            transform=build_transform(config.DATA.DATASET1, is_train=True, config=config),
            # transform1=build_transform_teacher(config.DATA.DATASET1, is_train=True, config=config),
        )
        test_dataset = TID2013Dataset6(
            config.DATA.DATA_PATH1,
            config.SET1.TEST_INDEX,
            config.DATA.PATCH_NUM1,
            transform=build_transform(config.DATA.DATASET1, is_train=False, config=config),
            # transform1=build_transform_teacher(is_train=False, config=config),
        )
    elif config.DATA.DATASET1 == "csiq":
        train_dataset = CSIQDataset6(
            config.DATA.DATA_PATH1,
            config.SET1.TRAIN_INDEX,
            config.DATA.PATCH_NUM1,
            transform=build_transform(config.DATA.DATASET1, is_train=True, config=config),
            # transform1=build_transform_teacher(is_train=True, config=config),
        )
        test_dataset = CSIQDataset6(
            config.DATA.DATA_PATH1,
            config.SET1.TEST_INDEX,
            config.DATA.PATCH_NUM1,
            transform=build_transform(config.DATA.DATASET1, is_train=False, config=config),
            # transform1=build_transform_teacher(is_train=False, config=config),
        )
    elif config.DATA.DATASET1 == "kadid":
        train_dataset = KADIDDataset6(
            config.DATA.DATA_PATH1,
            config.SET1.TRAIN_INDEX,
            config.DATA.PATCH_NUM1,
            transform=build_transform(config.DATA.DATASET1, is_train=True, config=config),
            # transform1=build_transform_teacher(is_train=True, config=config),
        )
        test_dataset = KADIDDataset6(
            config.DATA.DATA_PATH1,
            config.SET1.TEST_INDEX,
            config.DATA.PATCH_NUM1,
            transform=build_transform(config.DATA.DATASET1, is_train=False, config=config),
            # transform1=build_transform_teacher(is_train=False, config=config),
        )
    elif config.DATA.DATASET1 == "spaq":
        train_dataset = SPAQDATASET6(
            config.DATA.DATA_PATH1,
            config.SET1.TRAIN_INDEX,
            config.DATA.PATCH_NUM1,
            transform=build_transform(config.DATA.DATASET1, is_train=True, config=config),
            # transform1=build_transform_teacher(is_train=True, config=config),
        )
        test_dataset = SPAQDATASET6(
            config.DATA.DATA_PATH1,
            config.SET1.TEST_INDEX,
            config.DATA.PATCH_NUM1,
            transform=build_transform(config.DATA.DATASET1, is_train=False, config=config),
            # transform1=build_transform_teacher(is_train=False, config=config),
        )
    elif config.DATA.DATASET1 == "livefb":
        train_dataset = FBLIVEFolder6(
            config.DATA.DATA_PATH1,
            config.SET1.TRAIN_INDEX,
            config.DATA.PATCH_NUM1,
            transform=build_transform(config.DATA.DATASET1, is_train=True, config=config),
            # transform1=build_transform_teacher(is_train=True, config=config),
        )
        test_dataset = FBLIVEFolder6(
            config.DATA.DATA_PATH1,
            config.SET1.TEST_INDEX,
            config.DATA.PATCH_NUM1,
            transform=build_transform(config.DATA.DATASET1, is_train=False, config=config),
            # transform1=build_transform_teacher(is_train=False, config=config),
        )
    else:
        raise NotImplementedError("We only support common IQA dataset Now.")

    return train_dataset, test_dataset


def IQA_build_loader_a_krr(config):
    config.defrost()
    dataset_train, dataset_val = build_IQA_dataset_a_krr(config)
    config.freeze()
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset"
    )
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset"
    )

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == "part":
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = IQAPatchDistributedSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    # data_loader_train_val = torch.utils.data.DataLoader(
    #     dataset_train,
    #     sampler=sampler_train,
    #     batch_size=config.DATA.BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=config.DATA.NUM_WORKERS,
    #     pin_memory=config.DATA.PIN_MEMORY,
    #     drop_last=False,
    # )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    # setup mixup / cutmix
    mixup_fn = None

    # return dataset_train, dataset_val, data_loader_train, data_loader_train_val, data_loader_val, mixup_fn
    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def IQA_build_loader_b_krr(config):
    config.defrost()
    dataset_train, dataset_val = build_IQA_dataset_b_krr(config)
    config.freeze()
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset"
    )
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset"
    )

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == "part":
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = IQAPatchDistributedSampler(dataset_val)

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train,
    #     sampler=sampler_train,
    #     batch_size=config.DATA.BATCH_SIZE1,
    #     num_workers=config.DATA.NUM_WORKERS,
    #     pin_memory=config.DATA.PIN_MEMORY,
    #     drop_last=True,
    # )

    # data_loader_train_val = torch.utils.data.DataLoader(
    #     dataset_train,
    #     sampler=sampler_train,
    #     batch_size=config.DATA.BATCH_SIZE1,
    #     shuffle=False,
    #     num_workers=config.DATA.NUM_WORKERS,
    #     pin_memory=config.DATA.PIN_MEMORY,
    #     drop_last=False,
    # )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE1,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    # setup mixup / cutmix
    mixup_fn = None

    # return dataset_train, dataset_val, data_loader_train, data_loader_train_val, data_loader_val, mixup_fn
    return dataset_val, data_loader_val, mixup_fn