import os

import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = [""]

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 32
# change1
_C.DATA.BATCH_SIZE1 = 16
# Path to dataset, could be overwritten by command line argument
# K10K: /data/qgy/IQA-Dataset/koinq-10k
# LIVEC: /home/qinguanyi/dataset/IQA/ChallengeDB_release
_C.DATA.DATA_PATH = "/home/pws/IQA/dataset/tid2013"
# change2
_C.DATA.DATA_PATH1 = "/home/pws/IQA/dataset/tid2013"
# Dataset name
_C.DATA.DATASET = "tid2013"
# change3
_C.DATA.DATASET1 = "csiq"
# Aug images
_C.DATA.PATCH_NUM = 25
# change
_C.DATA.PATCH_NUM1 = 25
# Input image size
_C.DATA.IMG_SIZE = 224
# Random crop image size
_C.DATA.CROP_SIZE = (224, 224)
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = "bicubic"
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = "part"
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8


# -----------------------------------------------------------------------------
# SET settings
# -----------------------------------------------------------------------------
_C.SET = CN()
# K10K: 10073 LIVEC: 1162
_C.SET.COUNT = 10073
_C.SET.TRAIN_INDEX = None
_C.SET.TEST_INDEX = None
# change4
_C.SET1 = CN()
# K10K: 10073 LIVEC: 1162
_C.SET1.COUNT = 10073
_C.SET1.TRAIN_INDEX = None
_C.SET1.TEST_INDEX = None
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = "swin"
# Model name
_C.MODEL.NAME = "swin_tiny_patch4_window7_224"
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ""
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ""
# change5
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME_STU = ""
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.0
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
_C.MODEL.SWIN.SCORE_HEAD = 16
_C.MODEL.SWIN.SCORE_DIM = 32
_C.MODEL.SWIN.PRETRAINED = False
_C.MODEL.SWIN.PRETRAINED_MODEL_PATH = ""

# Swin Transformer V2 parameters
_C.MODEL.SWINV2 = CN()
_C.MODEL.SWINV2.PATCH_SIZE = 4
_C.MODEL.SWINV2.IN_CHANS = 3
_C.MODEL.SWINV2.EMBED_DIM = 96
_C.MODEL.SWINV2.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWINV2.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWINV2.WINDOW_SIZE = 7
_C.MODEL.SWINV2.MLP_RATIO = 4.0
_C.MODEL.SWINV2.QKV_BIAS = True
_C.MODEL.SWINV2.APE = False
_C.MODEL.SWINV2.PATCH_NORM = True
_C.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]

# Swin MLP parameters
_C.MODEL.SWIN_MLP = CN()
_C.MODEL.SWIN_MLP.PATCH_SIZE = 4
_C.MODEL.SWIN_MLP.IN_CHANS = 3
_C.MODEL.SWIN_MLP.EMBED_DIM = 96
_C.MODEL.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MLP.WINDOW_SIZE = 7
_C.MODEL.SWIN_MLP.MLP_RATIO = 4.0
_C.MODEL.SWIN_MLP.APE = False
_C.MODEL.SWIN_MLP.PATCH_NORM = True


# ConvNeXt parameters
_C.MODEL.CONV_NEXT = CN()
_C.MODEL.CONV_NEXT.IN_CHANS = 3
_C.MODEL.CONV_NEXT.DIM = [96, 192, 384, 768]
_C.MODEL.CONV_NEXT.DEPTHS = [3, 3, 9, 3]
_C.MODEL.CONV_NEXT.LAYER_SCALE_INIT = 1e-6
_C.MODEL.CONV_NEXT.HEAD_INIT_SCALE = 1.0
_C.MODEL.CONV_NEXT.SCORE_HEAD = 16
_C.MODEL.CONV_NEXT.SCORE_DIM = 32
_C.MODEL.CONV_NEXT.PRETRAINED = False
_C.MODEL.CONV_NEXT.PRETRAINED_MODEL_PATH = ""

# DeiT III
_C.MODEL.VIT = CN()
_C.MODEL.VIT.PATCH_SIZE = 16
_C.MODEL.VIT.EMBED_DIM = 384
_C.MODEL.VIT.DEPTH = 12
_C.MODEL.VIT.NUM_HEADS = 6
_C.MODEL.VIT.MLP_RATIO = 4
_C.MODEL.VIT.QKV_BIAS = True
_C.MODEL.VIT.PRETRAINED = True
_C.MODEL.VIT.PRETRAINED_MODEL_PATH = ""
# change6
_C.MODEL.VIT.PRETRAINED_MODEL_PATH_TEACHER = ""

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
# change7
_C.TRAIN.START_EPOCH_STU = 0
_C.TRAIN.EPOCHS_STU = 6
# change8
_C.TRAIN.START_EPOCH_STU1 = 0
_C.TRAIN.EPOCHS_STU1 = 9

_C.TRAIN.WARMUP_EPOCHS = 20
# change9
_C.TRAIN.WARMUP_EPOCHS_STU = 2
_C.TRAIN.WARMUP_EPOCHS_STU1 = 3

# change10
# incremental times
_C.DATA.SLICE_NUMBER = 6
# examplar step
# change11
_C.DATA.EXAMPLAR_STEP = 50
_C.DATA.EXAMPLAR_STEP1 = 50
_C.DATA.num_cluster = 1


_C.TRAIN.WEIGHT_DECAY = 0.05
# change12
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.BASE_LR1 = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
_C.TRAIN.WARMUP_LR1 = 5e-7
_C.TRAIN.MIN_LR1 = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint teacher
_C.TRAIN.AUTO_RESUME = True
# Auto resume from latest checkpoint student
_C.TRAIN.AUTO_RESUME_STU = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = "cosine"
# change
_C.TRAIN.LR_SCHEDULER1 = CN()
_C.TRAIN.LR_SCHEDULER1.NAME = "cosine"
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# change14
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS_STU = 30
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS_STU1 = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
_C.TRAIN.LR_SCHEDULER.DECAY_RATE1 = 0.5

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = "rand-m9-mstd0.5-inc1"
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = "pixel"
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = "batch"

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ""
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ""
# Tag of experiment, overwritten by command line argument
_C.TAG = "default"
# Frequency to save checkpoint
_C.SAVE_FREQ = 2
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Use tensorboard to track the trainning log
_C.TENSORBOARD = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# Use torchinfo to show the flow
_C.DEBUG_MODE = False
# Repeat exps for publications
_C.EXP_INDEX = 0
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0

_C.CLUSTER = True


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print("=> merge config from {}".format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args, local_rank):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.batch_size1:
        config.DATA.BATCH_SIZE1 = args.batch_size1
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.data_path1:
        config.DATA.DATA_PATH1 = args.data_path1
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == "O0":
            config.AMP_ENABLE = False
    if args.disable_amp:
        config.AMP_ENABLE = False
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.tensorboard:
        config.TENSORBOARD = True
    if args.throughput:
        config.THROUGHPUT_MODE = True
    if args.debug:
        config.DEBUG_MODE = True

    # set local rank for distributed training
    config.LOCAL_RANK = local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args, local_rank):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args, local_rank)

    return config
