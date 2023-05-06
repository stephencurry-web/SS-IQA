import argparse
import copy
import datetime
import json
import logging
import os
import pickle
import random
import time
import math

import heapq
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from scipy import stats
from timm.utils import AverageMeter  # accuracy
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision.transforms import transforms

from IQA import IQA_build_loader, IQA_build_loader2, IQA_build_loader3, IQA_build_loader5, SubsetRandomSampler
from config import get_config
from logger import create_logger
from lr_scheduler import build_scheduler, build_scheduler1, build_scheduler2, build_scheduler3
from models import build_model
from optimizer import build_optimizer, build_optimizer1
from utils import (
    NativeScalerWithGradNormCount,
    auto_resume_helper,
    load_checkpoint,
    load_pretrained,
    reduce_tensor,
    save_checkpoint,
    save_checkpoint1
)

def parse_option():
    parser = argparse.ArgumentParser(
        "Swin Transformer training and evaluation script", add_help=False
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--batch-size1", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--data-path1", type=str, help="path to dataset1")
    parser.add_argument(
        "--zip",
        action="store_true",
        help="use zipped dataset instead of folder dataset",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="part",
        choices=["no", "full", "part"],
        help="no: no cache, "
        "full: cache all data, "
        "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
    )
    parser.add_argument(
        "--pretrained",
        help="pretrained weight from checkpoint, could be imagenet22k pretrained weight",
    )
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument(
        "--accumulation-steps", type=int, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Using tensorboard to track the process",
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--disable_amp", action="store_true", help="Disable pytorch amp"
    )
    parser.add_argument(
        "--amp-opt-level",
        type=str,
        choices=["O0", "O1", "O2"],
        help="mixed precision opt level, if O0, no amp is used (deprecated!)",
    )
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use torchinfo to show the flow of tensor in model",
    )
    parser.add_argument(
        "--repeat", action="store_true", help="Test model for publications"
    )
    parser.add_argument("--rnum", type=int, help="Repeat num")
    # distributed training
    # os.environ['LOCAL_RANK'] = '1'  # 其中key和value均为string类型
    # os.putenv('LOCAL_RANK', '1')

    # os.environ.setdefault('LOCAL_RANK', '0')
    # os.environ["LOCAL_RANK"] = '0'
    # os.environ["RANK"] = '0'
    # os.environ["WORLD_SIZE"] = '1'
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '25627'
    local_rank = int(os.environ["LOCAL_RANK"])
    args, unparsed = parser.parse_known_args()

    config = get_config(args, local_rank)
    return args, config


def main(config):

    if dist.get_rank() == 0:
        group_name = config.TAG
        wandb_name = group_name + "_" + str(config.EXP_INDEX)
        wandb_dir = os.path.join(config.OUTPUT, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_runner = wandb.init(
            project="new",
            entity="wenshengpan",
            group=group_name,
            name=wandb_name,
            config={
                "epochs": config.TRAIN.EPOCHS,
                "batch_size": config.DATA.BATCH_SIZE,
                "batch_size1": config.DATA.BATCH_SIZE1,
                "patch_num": config.DATA.PATCH_NUM,
                "patch_num1": config.DATA.PATCH_NUM1,
                "EPOCHS_STU1": config.TRAIN.EPOCHS_STU1,
                "WARMUP_EPOCHS_STU1": config.TRAIN.WARMUP_EPOCHS_STU1,
                "BASE_LR1": config.TRAIN.BASE_LR1,
                "WARMUP_LR1": config.TRAIN.WARMUP_LR1,
                "MIN_LR1": config.TRAIN.MIN_LR1,
                "DECAY_RATE1": config.TRAIN.LR_SCHEDULER.DECAY_RATE1,
                "DECAY_EPOCHS_STU1": config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS_STU1,
                "SLICE_NUMBER": config.DATA.SLICE_NUMBER,
                "DATA.EXAMPLAR_STEP": config.DATA.EXAMPLAR_STEP,
                "DATA.EXAMPLAR_STEP1": config.DATA.EXAMPLAR_STEP1
            },
            dir=wandb_dir,
            reinit=True,
        )
        # wandb_runner.log({"Validating SRCC": 0.0, "Validating PLCC": 0.0, "Epoch": 0})
    else:
        wandb_runner = None

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    logger.info(f"cfg:{args.cfg}")
    s_model = build_model(config)
    s_model.cuda()
    s_model_without_ddp = s_model
    # logger.info(str(s_model))

    # t_model = build_model(config)
    # checkpoint = torch.load(config.MODEL.VIT.PRETRAINED_MODEL_PATH_TEACHER, map_location="cpu")
    # t_model.load_state_dict(checkpoint["model"])
    # logger.info(str(t_model))

    n_parameters = sum(p.numel() for p in s_model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(s_model, "flops"):
        flops = s_model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    print("s_model")

    criterion = torch.nn.SmoothL1Loss()
    examplar_total = []
    # dataset A
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = IQA_build_loader(config, 1)

    data_loader_train_ex1 = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
    )
    # dataset B
    dataset_train_total, dataset_val_total, data_loader_train_total, data_loader_val_total, mixup_fn \
        = IQA_build_loader2(config, 0)

    if dist.get_rank() == 0:
        logger.info(f"len(dataset_train):{len(dataset_train)}")
        logger.info(f"len(dataset_val):{len(dataset_val)}")
        logger.info(f"len(data_loader_train):{len(data_loader_train)}")
        logger.info(f"len(data_loader_val):{len(data_loader_val)}")
        logger.info(f"len(dataset_train_total):{len(dataset_train_total)}")
        logger.info(f"len(dataset_val_total):{len(dataset_val_total)}")
        logger.info(f"len(data_loader_train_total):{len(data_loader_train_total)}")
        logger.info(f"len(data_loader_val_total):{len(data_loader_val_total)}")

    optimizer = build_optimizer(config, s_model)
    s_model = torch.nn.parallel.DistributedDataParallel(
        s_model,
        device_ids=[config.LOCAL_RANK],
        broadcast_buffers=False,
        # find_unused_parameters=True,
    )

    if dist.get_rank() == 0:
        wandb_runner.watch(s_model)

    loss_scaler1 = NativeScalerWithGradNormCount()
    loss_scaler2 = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler1(
            config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS
        )
    else:
        lr_scheduler = build_scheduler1(config, optimizer, len(data_loader_train))

    if config.TENSORBOARD:
        writer = SummaryWriter(log_dir=config.OUTPUT)
    else:
        writer = None

    max_plcc_1_a = 0.0
    max_srcc_1_a = 0.0
    max_plcc_1_b = 0.0
    max_srcc_1_b = 0.0

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH_STU, config.TRAIN.EPOCHS_STU):
        data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch(config, s_model, criterion, data_loader_train,
            optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler1, writer, wandb_runner)

        # if dist.get_rank() == 0 and (
        #     epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)
        # ):
        #     save_checkpoint(config, epoch, s_model_without_ddp, max_plcc_1_a, optimizer,
        #         lr_scheduler, loss_scaler1, logger)

        srcc_1_a, plcc_1_a, loss_1_a = validate(config, data_loader_val, s_model, epoch, 1,
                                                                     len(dataset_val), writer, wandb_runner)
        # if config.TENSORBOARD == True:
        #     writer.add_scalars(
        #         "Validate Metrics",
        #         {"srcc_1": srcc_1_a, "plcc_1": plcc_1_a},
        #         epoch,
        #     )
        if dist.get_rank() == 0:
            wandb_runner.log(
                {"Validating srcc1": srcc_1_a, "Validating plcc1": plcc_1_a, "Epoch": epoch}
            )
        logger.info(
            f"PLCC and SRCC of the network on the {len(dataset_val)} test images: {plcc_1_a:.6f}, {srcc_1_a:.6f}"
        )
        if plcc_1_a >= max_plcc_1_a:
            max_plcc_1_a = max(max_plcc_1_a, plcc_1_a)
            max_srcc_1_a = srcc_1_a
        elif plcc_1_a < 0:
            max_srcc_1_a = 0
        logger.info(f"Max plcc_1: {max_plcc_1_a:.6f} Max srcc_1: {max_srcc_1_a:.6f}")
        if dist.get_rank() == 0:
            wandb_runner.summary["Best plcc_1"], wandb_runner.summary["Best srcc_1"] = (
                max_plcc_1_a,
                max_srcc_1_a,
            )

        srcc_1_b, plcc_1_b, loss_1_b = validate1(config, data_loader_val_total, s_model, epoch, 2,
                                                               len(dataset_val_total), writer, wandb_runner)
        if dist.get_rank() == 0:
            wandb_runner.log(
                {"Validating srcc2": srcc_1_b, "Validating plcc2": plcc_1_b, "Epoch": epoch}
            )
        logger.info(
            f"PLCC and SRCC of the network on the {len(dataset_val_total)} test images: {plcc_1_b:.6f}, {srcc_1_b:.6f}"
        )
        if plcc_1_b >= max_plcc_1_b:
            max_plcc_1_b = max(max_plcc_1_b, plcc_1_b)
            max_srcc_1_b = srcc_1_b
        elif plcc_1_b < 0:
            max_srcc_1_b = 0
        logger.info(f"Max plcc_2: {max_plcc_1_b:.6f} Max srcc_2: {max_srcc_1_b:.6f}")
        if dist.get_rank() == 0:
            wandb_runner.summary["Best plcc_2"], wandb_runner.summary["Best srcc_2"] = (
                max_plcc_1_b,
                max_srcc_1_b,
            )

    # copy to teacher
    t_model = copy.deepcopy(s_model)
    n_parameters = sum(p.numel() for p in t_model.parameters() if p.requires_grad)
    logger.info(f"tmodel number of params: {n_parameters}")
    if hasattr(t_model, "flops"):
        flops =t_model.flops()
        logger.info(f"tmodel number of GFLOPs: {flops / 1e9}")

    t_model.cuda()
    t_model_without_ddp = t_model
    t_model = torch.nn.parallel.DistributedDataParallel(
        t_model,
        device_ids=[config.LOCAL_RANK],
        broadcast_buffers=False,
        # find_unused_parameters=True,
    )
    logger.info("successful copy")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("data a Training time {}".format(total_time_str))
    if dist.get_rank() == 0:
        wandb_runner.alert(
            title="Run Finished",
            text=f"Max plcc_1_a: {max_plcc_1_a:.6f} Max srcc_1_a: {max_srcc_1_a:.6f} "
                 f"Max plcc_1_b: {max_plcc_1_b:.6f} Max srcc_1_b: {max_srcc_1_b:.6f} "
                 f"Training time: {total_time_str}",
        )

    # examplar_total = get_examplar(examplar_total, dataset_train)
    examplar_total = get_examplar1(examplar_total, dataset_train, data_loader_train_ex1, s_model, criterion)

    if dist.get_rank() == 0:
        logger.info(f"len(examplar_total):{len(examplar_total)}")

    # B已经训练过的数据
    train_set = []
    val_set = []

    for slice_num in range(config.DATA.SLICE_NUMBER):
        train_index, val_index = [], []
        for i, index in enumerate(config.SET1.TRAIN_INDEX):
            if(i % config.DATA.SLICE_NUMBER == slice_num):
                train_index.append(index)
        # logger.info(f"train_index:{train_index}")
        for i, index in enumerate(config.SET1.TEST_INDEX):
            if(i % config.DATA.SLICE_NUMBER == slice_num):
                val_index.append(index)
        # logger.info(f"val_index:{val_index}")
        assert len(val_index) != 0

        # for train
        dataset_train1, dataset_val1, data_loader_train1, data_loader_val1, mixup_fn \
                = IQA_build_loader3(config, train_index, val_index, examplar_total, 0)
        # for select examplar
        dataset_train2, dataset_val2,  data_loader_train2, data_loader_val2, mixup_fn \
            = IQA_build_loader5(config, train_index, val_index, train_set, val_set, 0)
        train_set = dataset_train2
        val_set = dataset_val2
        data_loader_train2_ex2 = torch.utils.data.DataLoader(
            dataset_train2,
            batch_size=1
        )

        optimizer2 = build_optimizer1(config, s_model)
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            lr_scheduler1 = build_scheduler2(
                config, optimizer2, len(data_loader_train1) // config.TRAIN.ACCUMULATION_STEPS
            )
        else:
            lr_scheduler1 = build_scheduler2(config, optimizer2, len(data_loader_train1))

        if dist.get_rank() == 0:

            logger.info(f"len(dataset_train1):{len(dataset_train1)}")
            logger.info(f"len(dataset_val1):{len(dataset_val1)}")
            logger.info(f"len(data_loader_train1):{len(data_loader_train1)}")
            logger.info(f"len(data_loader_val1):{len(data_loader_val1)}")
            logger.info(f"len(dataset_train2):{len(dataset_train2)}")
            logger.info(f"len(dataset_val2):{len(dataset_val2)}")
            logger.info(f"len(data_loader_train2):{len(data_loader_train2)}")
            logger.info(f"len(data_loader_val2):{len(data_loader_val2)}")

        # dataset A val set test
        max_plcc_2_a = 0.0
        max_srcc_2_a = 0.0
        
        # dataset B val set test
        max_plcc_2_b = 0.0
        max_srcc_2_b = 0.0

        for epoch in range(config.TRAIN.START_EPOCH_STU1, config.TRAIN.EPOCHS_STU1):
            data_loader_train1.sampler.set_epoch(epoch)

            train_one_epoch2(config, s_model, t_model, criterion, data_loader_train1, slice_num,
                            optimizer2, epoch, mixup_fn, lr_scheduler1, loss_scaler2, writer, wandb_runner)


            srcc_2_a, plcc_2_a, loss_2_a = validate2(config, data_loader_val, s_model, epoch, slice_num, 1,
                                                  len(dataset_val), writer, wandb_runner)
            srcc_2_b, plcc_2_b, loss_2_b = validate3(config, data_loader_val_total, s_model, epoch, slice_num, 2,
                                                     len(dataset_val_total), writer, wandb_runner)


            if dist.get_rank() == 0:
                wandb_runner.log(
                    {f"slice{slice_num}Validating SRCC1": srcc_2_a,
                     f"slice{slice_num}Validating PLCC1": plcc_2_a,
                     f"slice{slice_num}Validating SRCC2": srcc_2_b,
                     f"slice{slice_num}Validating PLCC2": plcc_2_b,
                     f"slice{slice_num}Epoch": epoch}
                )
            logger.info(
                f"slice{slice_num} PLCC1 and SRCC1 of the network on the {len(dataset_val)} test images: {plcc_2_a:.6f}, {srcc_2_a:.6f}; "
            )
            logger.info(
                f"slice{slice_num}PLCC2 and SRCC2 of the network on the {len(dataset_val_total)} test images: {plcc_2_b:.6f}, {srcc_2_b:.6f}"
            )
            if plcc_2_a >= max_plcc_2_a:
                max_plcc_2_a = max(max_plcc_2_a, plcc_2_a)
                max_srcc_2_a = srcc_2_a
            elif plcc_2_a < 0:
                max_srcc_2_a = 0
            logger.info(f"slice{slice_num} Max plcc_2_a: {max_plcc_2_a:.6f} Max srcc_2_a: {max_srcc_2_a:.6f}")
            if dist.get_rank() == 0:
                wandb_runner.summary["Best plcc_2_a"], wandb_runner.summary["Best srcc_2_a"] = (
                    max_plcc_2_a,
                    max_srcc_2_a,
                )

            if plcc_2_b >= max_plcc_2_b:
                max_plcc_2_b = max(max_plcc_2_b, plcc_2_b)
                max_srcc_2_b = srcc_2_b
            elif plcc_2_b < 0:
                max_srcc_2_b = 0
            logger.info(f"slice{slice_num} Max plcc_2_b: {max_plcc_2_b:.6f} Max srcc_2_b: {max_srcc_2_b:.6f}")
            if dist.get_rank() == 0:
                wandb_runner.summary["Best plcc_2_b"], wandb_runner.summary["Best srcc_2_b"] = (
                    max_plcc_2_b,
                    max_srcc_2_b,
                )


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("data b slice{} Training time {}".format(slice_num, total_time_str))
        if dist.get_rank() == 0:
            wandb_runner.alert(
                title="Run Finished",
                text=f" Max plcc_2_a: {max_plcc_2_a:.6f} Max srcc_2_a: {max_srcc_2_a:.6f} "
                     f" Max plcc_2_b: {max_plcc_2_b:.6f} Max srcc_2_b: {max_srcc_2_b:.6f}"
                     f" Training time: {total_time_str}",
            )
        # examplar_total = get_examplar(examplar_total, dataset_train1)
        if slice_num < config.DATA.SLICE_NUMBER-1:
            examplar_total = []
            # dataA examplar
            examplar_total = get_examplar1(examplar_total, dataset_train, data_loader_train_ex1, s_model, criterion)
            # dataB examplar
            examplar_total = get_examplar2(examplar_total, dataset_train2, data_loader_train2_ex2, s_model, t_model, criterion)
        if dist.get_rank() == 0:
            logger.info(f"len(examplar_total):{len(examplar_total)}")

    writer.close()
    if dist.get_rank() == 0:
        wandb_runner.finish()
        logging.shutdown()
    else:
        logging.shutdown()
    dist.barrier()
    return plcc_1_a, srcc_1_a, plcc_1_b, srcc_1_b, \
           plcc_2_a, srcc_2_a, plcc_2_b, srcc_2_b

# random select
def get_examplar(examplar, dataset_train):
    for idx, item in enumerate(dataset_train):
        if(idx % config.DATA.EXAMPLAR_STEP == 0):
            examplar.append(item)
    return examplar


def get_examplar1(examplar, dataset_train, data_loader_train_cv, s_model, criterion):
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    s_model.eval()
    top_k = math.ceil(math.ceil(len(data_loader_train_cv) / config.DATA.EXAMPLAR_STEP))
    print(top_k)
    point_array = []
    for idx, (image, target, _) in enumerate(data_loader_train_cv):
        with torch.no_grad():
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target.unsqueeze_(dim=-1)
            # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = s_model(image)
            point = criterion(output, target)
            if idx == 0:
                print(output.shape)
                print(target.shape)
                print(point)
            # point = abs(output - target)
            point_array.append(point.item())
    # print("topk begin")
    # print(point_array)

    top_k_index = heapq.nlargest(top_k, range(len(point_array)), point_array.__getitem__)
    for idx, item in enumerate(dataset_train):
        if idx in top_k_index:
            examplar.append(item)
    print("topk finish")
    return examplar


def get_examplar2(examplar, dataset_train, data_loader_train_cv, s_model, t_model, criterion):
    top_k = math.ceil(math.ceil(len(data_loader_train_cv) / config.DATA.EXAMPLAR_STEP1))
    print(top_k)
    point_array = []
    for idx, (image, _, _) in enumerate(data_loader_train_cv):
        with torch.no_grad():
            image = image.cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)
            # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = s_model(image)
            target = t_model(image)
            point = criterion(output, target)
            if idx == 0:
                print(output.shape)
                print(target.shape)
                print(point)
            # point = abs(output - target)
            point_array.append(point.item())
    # print("topk begin")
    # print(point_array)
    top_k_index = heapq.nlargest(top_k, range(len(point_array)), point_array.__getitem__)
    for idx, item in enumerate(dataset_train):
        if idx in top_k_index:
            examplar.append(item)
    print("topk finish")
    return examplar


def train_one_epoch2(
    config,
    s_model,
    t_model,
    criterion,
    data_loader,
    slice_num,
    optimizer,
    epoch,
    mixup_fn,
    lr_scheduler,
    loss_scaler,
    tensorboard=None,
    wandb_runner=None,
):
    if config.TENSORBOARD == True:
        assert tensorboard != None
    s_model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    # aux_loss_meter = AverageMeter()
    # aux_loss_fusl_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    pred_scores = []
    gt_scores = []
    start = time.time()
    end = time.time()
    for idx, (samples, target, data_num) in enumerate(data_loader):
        # if idx == 0:
        samples = samples.cuda(non_blocking=True)
        # mark = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        # if data_num.equal(mark):
        #     targets = target
        # else:
        #     targets = t_model(samples)
        # if idx == 0:
        #     print(samples.shape)
        #     print(target.shape)
        #     print(data_num.shape)
        with torch.no_grad():
            targets = t_model(samples)
        # if idx == 0:
        #     print(targets)
        #     print(target)
        data_num = data_num.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # targets = torch.where(data_num > 0, target, target_t)
        # count = 0
        for i, num in enumerate(data_num):
            if num.item() > 0:
                # count = count + 1
                # print("target change")
                # print(targets[i])
                # print(target[i])
                targets[i] = target[i]

        # print(count)
        # if idx == 0:
        #     print(targets)
        #     print(target)

        targets = targets.cuda(non_blocking=True)
        if mixup_fn is not None:
             samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            # outputs, aux_score = model(samples)
            outputs = s_model(samples)

        # targets.unsqueeze_(dim=-1)
        loss = criterion(outputs, targets)
        # aux_loss = criterion(aux_score, targets)
        # aux_loss_fusl = criterion(outputs + aux_score, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS
        # pred_scores = pred_scores + (outputs + aux_score).squeeze().cpu().tolist()
        pred_scores = pred_scores + (outputs).squeeze().cpu().tolist()
        gt_scores = gt_scores + targets.squeeze().cpu().tolist()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        # TODO: Too see if weights help the performace
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=s_model.parameters(),
            create_graph=is_second_order,
            update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0,
        )
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS
            )
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        # aux_loss_meter.update(aux_loss.item(), targets.size(0))
        # aux_loss_fusl_meter.update(aux_loss_fusl.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            # if config.TENSORBOARD == True:
            #     tensorboard.add_scalars(
            #         f"slice{slice_num}Training Loss",
            #         {"Loss": loss_meter.val},
            #         epoch * len(data_loader) + idx,
            #     )
            lr = optimizer.param_groups[0]["lr"]
            wd = optimizer.param_groups[0]["weight_decay"]
            if wandb_runner:
                wandb_runner.log(
                    {
                        f"slice{slice_num}Training Loss": loss_meter.val,
                        f"slice{slice_num}Learning Rate": lr,
                        f"slice{slice_num}Batch": epoch * len(data_loader) + idx,
                        f"slice{slice_num}Epoch": epoch
                    },
                )
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"slice{slice_num}Train: [{epoch+1}/{config.TRAIN.EPOCHS_STU1}][{idx}/{num_steps}]\t"
                f"slice{slice_num}eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t"
                f"slice{slice_num}time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"slice{slice_num}loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                # f"aux_loss {aux_loss_meter.val:.4f} ({aux_loss_meter.avg:.4f})\t"
                # f"aux_loss_fusl {aux_loss_fusl_meter.val:.4f} ({aux_loss_fusl_meter.avg:.4f})\t"
                f"slice{slice_num}grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"slice{slice_num}loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t"
                f"slice{slice_num}mem {memory_used:.0f}MB"
            )
    epoch_time = time.time() - start
    train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    if config.TENSORBOARD == True:
        tensorboard.add_scalar(
            f"slice{slice_num}Training SRCC",
            train_srcc,
            epoch,
        )
    logger.info(
        f"slice{slice_num}EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
    )
    logger.info(f"slice{slice_num}EPOCH {epoch} training SRCC: {train_srcc}")


@torch.no_grad()
def validate2(
    config, data_loader, s_model, epoch, slice_num, validate_num, val_len, tensorboard=None, wandb_runner=None
):
    criterion = torch.nn.SmoothL1Loss()
    s_model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    temp_pred_scores = []
    temp_gt_scores = []
    end = time.time()
    for idx, (images, target, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            # output, _ = model(images)
            output = s_model(images)
        target.unsqueeze_(dim=-1)
        temp_pred_scores.append(output.view(-1))
        temp_gt_scores.append(target.view(-1))
        # measure accuracy and record loss
        loss = criterion(output, target)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            # if config.TENSORBOARD == True and tensorboard != None:
            #     tensorboard.add_scalar(
            #         f"slice{slice_num}Validating Loss{validate_num}",
            #         loss_meter.val,
            #         epoch * len(data_loader) + idx,
            #     )
            if wandb_runner:
                wandb_runner.log(
                    {
                        f"slice{slice_num}Validating Loss{validate_num}": loss_meter.val,
                        f"slice{slice_num}Validate Batch{validate_num}": epoch * len(data_loader) + idx,
                    }
                )
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"slice{slice_num}Test: [{idx}/{len(data_loader)}]\t"
                f"slice{slice_num}Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"slice{slice_num}Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"slice{slice_num}Mem {memory_used:.0f}MB"
            )
    pred_scores = torch.cat(temp_pred_scores)
    gt_scores = torch.cat(temp_gt_scores)
    # For distributed parallel, collect all data and then run metrics.
    if torch.distributed.is_initialized():
        preds_gather_list = [
            torch.zeros_like(pred_scores) for _ in range(dist.get_world_size())
        ]
        torch.distributed.all_gather(preds_gather_list, pred_scores)
        gather_preds = torch.cat(preds_gather_list, dim=0)[:val_len]
        gather_preds = (
            (gather_preds.view(-1, config.DATA.PATCH_NUM)).mean(dim=-1)
        ).squeeze()
        grotruth_gather_list = [
            torch.zeros_like(gt_scores) for _ in range(dist.get_world_size())
        ]
        torch.distributed.all_gather(grotruth_gather_list, gt_scores)
        gather_grotruth = torch.cat(grotruth_gather_list, dim=0)[:val_len]
        gather_grotruth = (
            (gather_grotruth.view(-1, config.DATA.PATCH_NUM)).mean(dim=-1)
        ).squeeze()
        # na = gather_preds.cpu().numpy()
        # df = pd.DataFrame(na)
        # df.to_csv(header=None, path_or_buf="gather_preds.csv", index=None)
        # nb = gather_grotruth.cpu().numpy()
        # dfb = pd.DataFrame(nb)
        # dfb.to_csv(header=None, path_or_buf="gather_grotruth.csv", index=None)
        final_preds = gather_preds.cpu().tolist()
        final_grotruth = gather_grotruth.cpu().tolist()

    test_srcc, _ = stats.spearmanr(final_preds, final_grotruth)
    test_plcc, _ = stats.pearsonr(final_preds, final_grotruth)
    logger.info(f"slice{slice_num} validate{validate_num} * PLCC@ {test_plcc:.6f} SRCC@ {test_srcc:.6f}")
    return test_srcc, test_plcc, loss_meter.avg

@torch.no_grad()
def validate3(
    config, data_loader, s_model, epoch, slice_num, validate_num, val_len, tensorboard=None, wandb_runner=None
):
    criterion = torch.nn.SmoothL1Loss()
    s_model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    temp_pred_scores = []
    temp_gt_scores = []
    end = time.time()
    for idx, (images, target, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            # output, _ = model(images)
            output = s_model(images)
        target.unsqueeze_(dim=-1)
        temp_pred_scores.append(output.view(-1))
        temp_gt_scores.append(target.view(-1))
        # measure accuracy and record loss
        loss = criterion(output, target)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            # if config.TENSORBOARD == True and tensorboard != None:
            #     tensorboard.add_scalar(
            #         f"slice{slice_num}Validating Loss{validate_num}",
            #         loss_meter.val,
            #         epoch * len(data_loader) + idx,
            #     )
            if wandb_runner:
                wandb_runner.log(
                    {
                        f"slice{slice_num}Validating Loss{validate_num}": loss_meter.val,
                        f"slice{slice_num}Validate Batch{validate_num}": epoch * len(data_loader) + idx,
                    }
                )
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"slice{slice_num}Test: [{idx}/{len(data_loader)}]\t"
                f"slice{slice_num}Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"slice{slice_num}Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"slice{slice_num}Mem {memory_used:.0f}MB"
            )
    pred_scores = torch.cat(temp_pred_scores)
    gt_scores = torch.cat(temp_gt_scores)
    # For distributed parallel, collect all data and then run metrics.
    if torch.distributed.is_initialized():
        preds_gather_list = [
            torch.zeros_like(pred_scores) for _ in range(dist.get_world_size())
        ]
        torch.distributed.all_gather(preds_gather_list, pred_scores)
        gather_preds = torch.cat(preds_gather_list, dim=0)[:val_len]
        gather_preds = (
            (gather_preds.view(-1, config.DATA.PATCH_NUM1)).mean(dim=-1)
        ).squeeze()
        grotruth_gather_list = [
            torch.zeros_like(gt_scores) for _ in range(dist.get_world_size())
        ]
        torch.distributed.all_gather(grotruth_gather_list, gt_scores)
        gather_grotruth = torch.cat(grotruth_gather_list, dim=0)[:val_len]
        gather_grotruth = (
            (gather_grotruth.view(-1, config.DATA.PATCH_NUM1)).mean(dim=-1)
        ).squeeze()
        # na = gather_preds.cpu().numpy()
        # df = pd.DataFrame(na)
        # df.to_csv(header=None, path_or_buf="gather_preds.csv", index=None)
        # nb = gather_grotruth.cpu().numpy()
        # dfb = pd.DataFrame(nb)
        # dfb.to_csv(header=None, path_or_buf="gather_grotruth.csv", index=None)
        final_preds = gather_preds.cpu().tolist()
        final_grotruth = gather_grotruth.cpu().tolist()

    test_srcc, _ = stats.spearmanr(final_preds, final_grotruth)
    test_plcc, _ = stats.pearsonr(final_preds, final_grotruth)
    logger.info(f"slice{slice_num} validate{validate_num} * PLCC@ {test_plcc:.6f} SRCC@ {test_srcc:.6f}")
    return test_srcc, test_plcc, loss_meter.avg

def train_one_epoch(
    config,
    s_model,
    criterion,
    data_loader,
    optimizer,
    epoch,
    mixup_fn,
    lr_scheduler,
    loss_scaler,
    tensorboard=None,
    wandb_runner=None,
):
    # examplar_img = []
    # examplar_label = []
    if config.TENSORBOARD == True:
        assert tensorboard != None
    s_model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    # aux_loss_meter = AverageMeter()
    # aux_loss_fusl_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    pred_scores = []
    gt_scores = []
    start = time.time()
    end = time.time()
    for idx, (samples, targets, _) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            # outputs, aux_score = model(samples)
            outputs = s_model(samples)
        targets.unsqueeze_(dim=-1)
        loss = criterion(outputs, targets)
        # aux_loss = criterion(aux_score, targets)
        # aux_loss_fusl = criterion(outputs + aux_score, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS
        # pred_scores = pred_scores + (outputs + aux_score).squeeze().cpu().tolist()
        pred_scores = pred_scores + (outputs).squeeze().cpu().tolist()
        gt_scores = gt_scores + targets.squeeze().cpu().tolist()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        # TODO: Too see if weights help the performace
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=s_model.parameters(),
            create_graph=is_second_order,
            update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0,
        )
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS
            )
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        loss_meter.update(loss.item(), targets.size(0))
        # aux_loss_meter.update(aux_loss.item(), targets.size(0))
        # aux_loss_fusl_meter.update(aux_loss_fusl.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            # if config.TENSORBOARD == True:
            #     tensorboard.add_scalars(
            #         "Training Loss",
            #         {"Loss": loss_meter.val},
            #         epoch * len(data_loader) + idx,
            #     )
            lr = optimizer.param_groups[0]["lr"]
            wd = optimizer.param_groups[0]["weight_decay"]
            if wandb_runner:
                wandb_runner.log(
                    {
                        "Training Loss": loss_meter.val,
                        "Learning Rate": lr,
                        "Batch": epoch * len(data_loader) + idx,
                        "Epoch": epoch
                    },
                )
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch+1}/{config.TRAIN.EPOCHS_STU}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                # f"aux_loss {aux_loss_meter.val:.4f} ({aux_loss_meter.avg:.4f})\t"
                # f"aux_loss_fusl {aux_loss_fusl_meter.val:.4f} ({aux_loss_fusl_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )
    epoch_time = time.time() - start
    train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    if config.TENSORBOARD == True:
        tensorboard.add_scalar(
            "Training SRCC",
            train_srcc,
            epoch,
        )
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
    )
    logger.info(f"EPOCH {epoch} training SRCC: {train_srcc}")
    # return examplar_img, examplar_label


@torch.no_grad()
def validate(
    config, data_loader, model, epoch, validate_num, val_len, tensorboard=None, wandb_runner=None
):
    criterion = torch.nn.SmoothL1Loss()
    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    temp_pred_scores = []
    temp_gt_scores = []
    end = time.time()
    for idx, (images, target, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            # output, _ = model(images)
            output = model(images)
        target.unsqueeze_(dim=-1)
        temp_pred_scores.append(output.view(-1))
        temp_gt_scores.append(target.view(-1))
        # measure accuracy and record loss
        loss = criterion(output, target)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            # if config.TENSORBOARD == True and tensorboard != None:
            #     tensorboard.add_scalar(
            #         f"Validating Loss{validate_num}",
            #         loss_meter.val,
            #         epoch * len(data_loader) + idx,
            #     )
            if wandb_runner:
                wandb_runner.log(
                    {
                        f"Validating Loss{validate_num}": loss_meter.val,
                        f"Validate Batch{validate_num}": epoch * len(data_loader) + idx,
                    }
                )
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"Mem {memory_used:.0f}MB"
            )
    pred_scores = torch.cat(temp_pred_scores)
    gt_scores = torch.cat(temp_gt_scores)
    # For distributed parallel, collect all data and then run metrics.
    if torch.distributed.is_initialized():
        preds_gather_list = [
            torch.zeros_like(pred_scores) for _ in range(dist.get_world_size())
        ]
        torch.distributed.all_gather(preds_gather_list, pred_scores)
        gather_preds = torch.cat(preds_gather_list, dim=0)[:val_len]
        gather_preds = (
            (gather_preds.view(-1, config.DATA.PATCH_NUM)).mean(dim=-1)
        ).squeeze()
        grotruth_gather_list = [
            torch.zeros_like(gt_scores) for _ in range(dist.get_world_size())
        ]
        torch.distributed.all_gather(grotruth_gather_list, gt_scores)
        gather_grotruth = torch.cat(grotruth_gather_list, dim=0)[:val_len]
        gather_grotruth = (
            (gather_grotruth.view(-1, config.DATA.PATCH_NUM)).mean(dim=-1)
        ).squeeze()
        # na = gather_preds.cpu().numpy()
        # df = pd.DataFrame(na)
        # df.to_csv(header=None, path_or_buf="gather_preds.csv", index=None)
        # nb = gather_grotruth.cpu().numpy()
        # dfb = pd.DataFrame(nb)
        # dfb.to_csv(header=None, path_or_buf="gather_grotruth.csv", index=None)
        final_preds = gather_preds.cpu().tolist()
        final_grotruth = gather_grotruth.cpu().tolist()

    test_srcc, _ = stats.spearmanr(final_preds, final_grotruth)
    test_plcc, _ = stats.pearsonr(final_preds, final_grotruth)
    logger.info(f" * PLCC@ {test_plcc:.6f} SRCC@ {test_srcc:.6f}")
    return test_srcc, test_plcc, loss_meter.avg

@torch.no_grad()
def validate1(
    config, data_loader, model, epoch, validate_num, val_len, tensorboard=None, wandb_runner=None
):
    criterion = torch.nn.SmoothL1Loss()
    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    temp_pred_scores = []
    temp_gt_scores = []
    end = time.time()
    for idx, (images, target, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            # output, _ = model(images)
            output = model(images)
        target.unsqueeze_(dim=-1)
        temp_pred_scores.append(output.view(-1))
        temp_gt_scores.append(target.view(-1))
        # measure accuracy and record loss
        loss = criterion(output, target)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            # if config.TENSORBOARD == True and tensorboard != None:
            #     tensorboard.add_scalar(
            #         f"Validating Loss{validate_num}",
            #         loss_meter.val,
            #         epoch * len(data_loader) + idx,
            #     )
            if wandb_runner:
                wandb_runner.log(
                    {
                        f"Validating Loss{validate_num}": loss_meter.val,
                        f"Validate Batch{validate_num}": epoch * len(data_loader) + idx,
                    }
                )
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"Mem {memory_used:.0f}MB"
            )
    pred_scores = torch.cat(temp_pred_scores)
    gt_scores = torch.cat(temp_gt_scores)
    # For distributed parallel, collect all data and then run metrics.
    if torch.distributed.is_initialized():
        preds_gather_list = [
            torch.zeros_like(pred_scores) for _ in range(dist.get_world_size())
        ]
        torch.distributed.all_gather(preds_gather_list, pred_scores)
        gather_preds = torch.cat(preds_gather_list, dim=0)[:val_len]
        gather_preds = (
            (gather_preds.view(-1, config.DATA.PATCH_NUM)).mean(dim=-1)
        ).squeeze()
        grotruth_gather_list = [
            torch.zeros_like(gt_scores) for _ in range(dist.get_world_size())
        ]
        torch.distributed.all_gather(grotruth_gather_list, gt_scores)
        gather_grotruth = torch.cat(grotruth_gather_list, dim=0)[:val_len]
        gather_grotruth = (
            (gather_grotruth.view(-1, config.DATA.PATCH_NUM)).mean(dim=-1)
        ).squeeze()
        # na = gather_preds.cpu().numpy()
        # df = pd.DataFrame(na)
        # df.to_csv(header=None, path_or_buf="gather_preds.csv", index=None)
        # nb = gather_grotruth.cpu().numpy()
        # dfb = pd.DataFrame(nb)
        # dfb.to_csv(header=None, path_or_buf="gather_grotruth.csv", index=None)
        final_preds = gather_preds.cpu().tolist()
        final_grotruth = gather_grotruth.cpu().tolist()

    test_srcc, _ = stats.spearmanr(final_preds, final_grotruth)
    test_plcc, _ = stats.pearsonr(final_preds, final_grotruth)
    logger.info(f" * PLCC@ {test_plcc:.6f} SRCC@ {test_srcc:.6f}")
    return test_srcc, test_plcc, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, target, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(
            f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}"
        )
        return


if __name__ == "__main__":
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    dist.barrier()

    if args.repeat:
        assert args.rnum > 1
        num = args.rnum
    else:
        num = 1
    base_path = config.OUTPUT
    logger = logging.getLogger(name=f"{config.MODEL.NAME}")
    plcc1_notrain_total = 0.0
    srcc1_notrain_total = 0.0
    plcc2_notrain_total = 0.0
    srcc2_notrain_total = 0.0
    plcc1_train_total = 0.0
    srcc1_train_total = 0.0
    plcc2_train_total = 0.0
    srcc2_train_total = 0.0
    for i in range(num):
        if num > 1:
            config.defrost()
            config.OUTPUT = os.path.join(base_path, str(i))
            config.EXP_INDEX = i + 1
            config.SET.TRAIN_INDEX = None
            config.SET.TEST_INDEX = None
            config.SET1.TRAIN_INDEX = None
            config.SET1.TEST_INDEX = None
            config.freeze()
        random.seed(None)

        os.makedirs(config.OUTPUT, exist_ok=True)

        filename = "sel_num.data"
        filename1 = "sel_num.data1"
        if dist.get_rank() == 0:
            sel_path = os.path.join(config.OUTPUT, filename)
            if not os.path.exists(sel_path):
                sel_num = list(range(0, config.SET.COUNT))
                random.shuffle(sel_num)
                with open(os.path.join(config.OUTPUT, filename), "wb") as f:
                    pickle.dump(sel_num, f)
                del sel_num

            sel_path1 = os.path.join(config.OUTPUT, filename1)
            if not os.path.exists(sel_path1):
                sel_num1 = list(range(0, config.SET1.COUNT))
                random.shuffle(sel_num1)
                with open(os.path.join(config.OUTPUT, filename1), "wb") as f:
                    pickle.dump(sel_num1, f)
                del sel_num1
        dist.barrier()

        with open(os.path.join(config.OUTPUT, filename), "rb") as f:
            sel_num = pickle.load(f)

        with open(os.path.join(config.OUTPUT, filename1), "rb") as f:
            sel_num1 = pickle.load(f)

        config.defrost()
        config.SET.TRAIN_INDEX = sel_num[0: int(round(0.8 * len(sel_num)))]
        config.SET.TEST_INDEX = sel_num[int(round(0.8 * len(sel_num))): len(sel_num)]

        config.SET1.TRAIN_INDEX = sel_num1[0: int(round(0.8 * len(sel_num1)))]
        # logger.info(f"config.SET1.TRAIN_INDEX:{config.SET1.TRAIN_INDEX}")
        config.SET1.TEST_INDEX = sel_num1[int(round(0.8 * len(sel_num1))): len(sel_num1)]
        # logger.info(f"config.SET1.TEST_INDEX:{config.SET1.TEST_INDEX}")
        config.freeze()

        seed = config.SEED + dist.get_rank()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.benchmark = True

        create_logger(
            logger,
            output_dir=config.OUTPUT,
            dist_rank=dist.get_rank(),
            name=f"{config.MODEL.NAME}",
        )

        if dist.get_rank() == 0:
            path = os.path.join(config.OUTPUT, "config.json")
            with open(path, "w") as f:
                f.write(config.dump())
            logger.info(f"Full config saved to {path}")

        # print config
        # logger.info(config.dump())
        # logger.info(json.dumps(vars(args)))
        plcc1_notrain, srcc1_notrain, plcc2_notrain, srcc2_notrain, plcc1_train, srcc1_train, plcc2_train, srcc2_train \
            = main(config)
        plcc1_notrain_total += plcc1_notrain
        srcc1_notrain_total += srcc1_notrain
        plcc2_notrain_total += plcc2_notrain
        srcc2_notrain_total += srcc2_notrain
        plcc1_train_total += plcc1_train
        srcc1_train_total += srcc1_train
        plcc2_train_total += plcc2_train
        srcc2_train_total += srcc2_train

        if i == num-1:
            plcc1_notrain_average = '%.4f' % (plcc1_notrain_total / num)
            srcc1_notrain_average = '%.4f' % (srcc1_notrain_total / num)
            plcc2_notrain_average = '%.4f' % (plcc2_notrain_total / num)
            srcc2_notrain_average = '%.4f' % (srcc2_notrain_total / num)
            plcc1_train_average = '%.4f' % (plcc1_train_total / num)
            srcc1_train_average = '%.4f' % (srcc1_train_total / num)
            plcc2_train_average = '%.4f' % (plcc2_train_total / num)
            srcc2_train_average = '%.4f' % (srcc2_train_total / num)
            improve1 = plcc1_train_average - plcc1_notrain_average
            improve2 = srcc1_train_average - srcc1_notrain_average
            improve3 = plcc2_train_average - plcc2_notrain_average
            improve4 = srcc2_train_average - srcc2_notrain_average
            logger.info(f"plcc1_notrain, srcc1_notrain:{plcc1_notrain_average},{srcc1_notrain_average};"
                        f"plcc2_notrain, srcc2_notrain:{plcc2_notrain_average},{srcc2_notrain_average};"
                        f"plcc1_train, srcc1_train:{plcc1_train_average},{srcc1_train_average};"
                        f"plcc2_train, srcc2_train:{plcc2_train_average},{srcc2_train_average}")
            logger.info(f"plcc1 improve:{improve1};"
                        f"srcc1 improve:{improve2};"
                        f"plcc2 improve:{improve3};"
                        f"srcc2 improve:{improve4}")
        logger.handlers.clear()
