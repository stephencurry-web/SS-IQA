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
from optimizer import build_optimizer
from lr_scheduler import build_scheduler2

from util import (
    NativeScalerWithGradNormCount,
    auto_resume_helper,
    load_checkpoint,
    load_pretrained,
    reduce_tensor,
    save_checkpoint,
    save_checkpoint1
)
from IQA import SubsetRandomSampler, IQAPatchDistributedSampler


# random select
# def get_random_examplar(examplar, dataset_train, examplar_step):
#     # for idx, item in enumerate(dataset_train):
#     #     if(idx % examplar_step == 0):
#     #         examplar.append(item)
#     leng = len(dataset_train) // examplar_step
#     examplar = random.sample(dataset_train.samples, leng)
#     # for idx in range(leng):
#     #     examplar.append(dataset_train[idx * examplar_step])
#     return examplar
def get_random_examplar(examplar, dataset_train, examplar_step):
    leng = len(dataset_train) // examplar_step
    for idx in range(leng):
        examplar.append(dataset_train[idx * examplar_step])
    return examplar


def get_A_examplar(examplar, dataset_train, data_loader_train_cv, s_model, examplar_step, logger):
    s_model.eval()
    top_k = math.ceil(math.ceil(len(dataset_train) / examplar_step))
    if dist.get_rank() == 0:
        logger.info(f"top_k:{top_k}")
    point_array = []
    for idx, (image, target) in enumerate(data_loader_train_cv):
        with torch.no_grad():
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target.unsqueeze_(dim=-1)
            # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output, _ = s_model(image)
            res = torch.sub(target, output)
            res = torch.abs(res)

            # if idx % 100 == 0:
            #      logger.info(f"res:{res}")
            # #     print(target.shape)

            for _, point in enumerate(res):
                point_array.append(point.item())

    logger.info("get point")
    top_k_index = heapq.nlargest(top_k, range(len(point_array)), point_array.__getitem__)
    # top_k_index = heapq.nsmallest(top_k, range(len(point_array)), point_array.__getitem__)
    # for idx, item in enumerate(dataset_train):
    #     if idx in top_k_index:
    #         examplar.append(item)
    for idx in top_k_index:
        examplar.append(dataset_train[idx])
    if dist.get_rank() == 0:
        logger.info("topk finish")
    return examplar


def get_B_examplar(examplar, dataset_train, data_loader_train_cv, s_model, t_model, examplar_step):
    s_model.eval()
    t_model.eval()
    top_k = math.ceil(math.ceil(len(dataset_train) / examplar_step))
    if dist.get_rank() == 0:
        print("top_k:", top_k)
    point_array = []
    for idx, (image, _) in enumerate(data_loader_train_cv):
        with torch.no_grad():
            image = image.cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)
            # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output, _ = s_model(image)
            target, _ = t_model(image)
            res = torch.sub(target, output)
            res = torch.abs(res)

            # if idx == 0:
            #     print(output.shape)
            #     print(target.shape)

            for _, point in enumerate(res):
                point_array.append(point.item())
    print("get point")
    top_k_index = heapq.nlargest(top_k, range(len(point_array)), point_array.__getitem__)
    for idx, item in enumerate(dataset_train):
        if idx in top_k_index:
            examplar.append(item)
    if dist.get_rank() == 0:
        print("topk finish")
    return examplar


def get_mnemonics(config, examplar, dataset_train, s_model, examplar_step, criterion, batch_size, logger):

    # 构建数据集 data_loader_train用来随机选择子集， data_loader_train_val用来做验证
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
        batch_size=batch_size,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )
    data_loader_train_val = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    # 挑选次数
    epochs = 50
    # 挑选比例
    selection_ratio = examplar_step  # select 1 sample for every 10 training samples
    # 初始化无穷大的验证loss
    best_val_loss = float('inf')
    # 取到最佳值的次数，5次就停止
    best_time = 0
    # 保存选择的样本的索引
    temp_num = []

    for epoch in range(epochs):
        total_samples = 0
        selected_samples = 0
        # 保存选择的样本的索引
        temp_num = []
        logger.info(f"Epoch:{epoch+1}")
        # 每个epoch重新copy模型
        c_model = copy.deepcopy(s_model)
        c_model = torch.nn.parallel.DistributedDataParallel(
            c_model,
            device_ids=[config.LOCAL_RANK],
            broadcast_buffers=False,
            # find_unused_parameters=True,
        )

        optimizer = build_optimizer(config, c_model)
        optimizer.zero_grad()
        c_model.train()

        loss_scaler = NativeScalerWithGradNormCount()
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        scaler_meter = AverageMeter()
        end = time.time()

        logger.info("start training")
        # iterate over batches of the training data
        for idx, (samples, targets, data_num) in enumerate(data_loader_train):
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            # randomly select whether to use this batch or skip it
            if torch.rand(1) < (selection_ratio - 1) / selection_ratio:
                total_samples += len(samples)
                continue

            # save index
            for idx in range(len(data_num)):
                temp_num.append(data_num[idx].item())

            total_samples += len(samples)
            selected_samples += len(samples)

            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                # outputs, aux_score = model(samples)
                outputs, feat = c_model(samples)
            targets.unsqueeze_(dim=-1)
            loss = criterion(outputs, targets)
            # aux_loss = criterion(aux_score, targets)
            # aux_loss_fusl = criterion(outputs + aux_score, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS

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

        logger.info("validate")
        val_loss = validate(config, data_loader_train_val, c_model)

        # check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_time += 1
            logger.info(f"Selected sample {selected_samples} / {total_samples}, Val loss improved to {best_val_loss:.4f}")
        else:
            logger.info(f"Selected sample {selected_samples} / {total_samples}, Val loss did not improve ({val_loss:.4f})")
        if best_time == 5:
            break

        del c_model
        del optimizer

    for idx, item in enumerate(dataset_train):
        if idx in temp_num:
            examplar.append(item)
    return examplar


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.SmoothL1Loss()
    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    temp_pred_scores = []
    temp_gt_scores = []
    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            # output, _ = model(images)
            output, feat = model(images)
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

    return loss_meter.avg


def get_A_examplar1(examplar, dataset_train, data_loader_train_cv, s_model, examplar_step, logger):
    s_model.eval()
    top_k = math.ceil(math.ceil(len(dataset_train) / examplar_step))
    if dist.get_rank() == 0:
        logger.info(f"top_k:{top_k}")
    point_array = []
    for idx, (image, target) in enumerate(data_loader_train_cv):
        with torch.no_grad():
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target.unsqueeze_(dim=-1)
            # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output, _ = s_model(image)
            res = torch.abs(target - output)
        del image, target, output
            #
            # if idx % 100 == 0:
            #     logger.info(f"res:{res}")

        for _, point in enumerate(res):
            point_array.append(point.item())

    logger.info("get point")
    # top_k_index = heapq.nlargest(top_k, range(len(point_array)), point_array.__getitem__)
    top_k_index = heapq.nsmallest(top_k, range(len(point_array)), point_array.__getitem__)
    # for idx, item in enumerate(dataset_train):
    #     if idx in top_k_index:
    #         examplar.append(item)
    logger.info("find index")
    # for idx in top_k_index:
    #     examplar.append(dataset_train[idx])

    batch_size = 100
    batch = []
    for idx in top_k_index:
        batch.append(dataset_train[idx])
        if len(batch) == batch_size:
            examplar += batch
            batch = []
    # Add the last batch if it is smaller than the batch size
    if batch:
        examplar += batch

    if dist.get_rank() == 0:
        logger.info("topk finish")
    return examplar


def get_B_examplar1(examplar, dataset_train, data_loader_train_cv, s_model, t_model, examplar_step):
    s_model.eval()
    t_model.eval()
    top_k = math.ceil(math.ceil(len(dataset_train) / examplar_step))
    if dist.get_rank() == 0:
        print(top_k)
    point_array = []
    for idx, (image, _) in enumerate(data_loader_train_cv):
        with torch.no_grad():
            image = image.cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)
            # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = s_model(image)
            target = t_model(image)
            res = torch.sub(target, output)
            res = torch.abs(res)
            #
            # if idx == 0:
            #     print(output.shape)
            #     print(target.shape)

            for _, point in enumerate(res):
                point_array.append(point.item())

    top_k_index = heapq.nsmallest(top_k, range(len(point_array)), point_array.__getitem__)
    for idx, item in enumerate(dataset_train):
        if idx in top_k_index:
            examplar.append(item)

    if dist.get_rank() == 0:
        print("topk finish")
    return examplar