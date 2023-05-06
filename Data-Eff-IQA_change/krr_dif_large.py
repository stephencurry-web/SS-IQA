import argparse
import copy
import datetime
import logging
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from scipy import stats
from sklearn.kernel_ridge import KernelRidge
from timm.utils import AverageMeter  # accuracy
from torch.utils.tensorboard import SummaryWriter

from IQA import (
    IQA_build_loader_knn,
    build_IQA_dataset_krr_student,
    build_IQA_dataset_krr_teacher,
    IQA_build_loader_a_krr,
    IQA_build_loader_b_krr,
)

from config import get_config
from logger import create_logger
from lr_scheduler import build_scheduler1, build_scheduler2, build_scheduler3
from models import build_model, build_model_pws1, get_krr_label, get_regressor
from optimizer import build_optimizer, build_optimizer1
from util import (
    NativeScalerWithGradNormCount,
    reduce_tensor,
    save_checkpoint,
    save_checkpoint1
)
from examplar import get_A_examplar1, get_B_examplar, get_mnemonics, get_random_examplar
from ub_teacher import _update_teacher_model

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
    # os.environ['MASTER_PORT'] = '25618'
    local_rank = int(os.environ["LOCAL_RANK"])
    args, unparsed = parser.parse_known_args()

    config = get_config(args, local_rank)
    return args, config


def main(config, save_num):

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
    
    # teacher model
    t_model = build_model_pws1(config)
    t_model.cuda()
    t_model_without_ddp = t_model

    # s_model = build_model_pws1(config)
    # s_model.cuda()
    # s_model_without_ddp = s_model

    n_parameters = sum(p.numel() for p in t_model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(t_model, "flops"):
        flops = t_model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")


    # # dataset A 训练集以及测试集
    # dataset_train_A, dataset_val_A, data_loader_train_A, data_loader_train_val_A, data_loader_val_A, mixup_fn_A = IQA_build_loader_a_knn_dif(
    #     config)
    # logger.info(f"{len(data_loader_train_val_A)}")

    dataset_train_A, dataset_val_A, data_loader_train_A, data_loader_val_A, mixup_fn_A = IQA_build_loader_a_krr(config)


    # dataset B 训练集以及测试集
    # dataset_train_B, dataset_val_B, data_loader_train_B, data_loader_train_val_B, data_loader_val_B, mixup_fn_B = IQA_build_loader_b_knn_dif(config)
    # logger.info(f"{len(data_loader_train_val_B)}")

    dataset_val_B, data_loader_val_B, mixup_fn_B = IQA_build_loader_b_krr(config)

    # 打印
    if dist.get_rank() == 0:
        logger.info(f"len(dataset_train_A):{len(dataset_train_A)}")
        logger.info(f"len(dataset_val_A):{len(dataset_val_A)}")
        logger.info(f"len(data_loader_train_A):{len(data_loader_train_A)}")
        logger.info(f"len(data_loader_val_A):{len(data_loader_val_A)}")
        # logger.info(f"len(dataset_train_B):{len(dataset_train_B)}")
        logger.info(f"len(dataset_val_B):{len(dataset_val_B)}")
        # logger.info(f"len(data_loader_train_B):{len(data_loader_train_B)}")
        logger.info(f"len(data_loader_val_B):{len(data_loader_val_B)}")

    # t_model的optimizer和lr_scheduler
    optimizer_A = build_optimizer(config, t_model)
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler_A = build_scheduler1(
            config, optimizer_A, len(data_loader_train_A) // config.TRAIN.ACCUMULATION_STEPS
        )
    else:
        lr_scheduler_A = build_scheduler1(config, optimizer_A, len(data_loader_train_A))

    t_model = torch.nn.parallel.DistributedDataParallel(
        t_model,
        device_ids=[config.LOCAL_RANK],
        broadcast_buffers=False,
        # find_unused_parameters=True,
    )

    if dist.get_rank() == 0:
        wandb_runner.watch(t_model)

    criterion = torch.nn.SmoothL1Loss()
    loss_scaler1 = NativeScalerWithGradNormCount()
    loss_scaler2 = NativeScalerWithGradNormCount()

    if config.TENSORBOARD:
        writer = SummaryWriter(log_dir=config.OUTPUT)
    else:
        writer = None

    max_plcc_2_a = 0.0
    max_srcc_2_a = 0.0
    max_plcc_2_b = 0.0
    max_srcc_2_b = 0.0


    logger.info("Start training")
    start_time = time.time()
    # 在 dataset A上训练teacher
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train_A.sampler.set_epoch(epoch)

        if dist.get_rank() == 0 and (
            epoch % config.SAVE_FREQ == 1 or epoch == (config.TRAIN.EPOCHS - 1)
        ):
            save_checkpoint(
                config,
                epoch,
                t_model_without_ddp,
                max_plcc_2_a,
                max_plcc_2_b,
                optimizer_A,
                lr_scheduler_A,
                loss_scaler1,
                logger,
                save_num
            )


        train_one_epoch(config, t_model, criterion, data_loader_train_A,
                        optimizer_A, epoch, mixup_fn_A, lr_scheduler_A, loss_scaler1, writer, wandb_runner)


        # dataset A测试
        srcc_2_a, plcc_2_a, loss_2_a = validate(config, data_loader_val_A, t_model, epoch, 1,
                                                len(dataset_val_A), writer, wandb_runner)

        if dist.get_rank() == 0:
            wandb_runner.log(
                {"Validating srcc1": srcc_2_a, "Validating plcc1": plcc_2_a, "Epoch": epoch}
            )
        logger.info(
            f"PLCC and SRCC of the network on the {len(dataset_val_A)} test images: {plcc_2_a:.6f}, {srcc_2_a:.6f}"
        )
        if plcc_2_a >= max_plcc_2_a:
            max_plcc_2_a = max(max_plcc_2_a, plcc_2_a)
            max_srcc_2_a = srcc_2_a
        elif plcc_2_a < 0:
            max_srcc_2_a = 0
        logger.info(f"Max plcc_1: {max_plcc_2_a:.6f} Max srcc_1: {max_srcc_2_a:.6f}")
        if dist.get_rank() == 0:
            wandb_runner.summary["Best plcc_1"], wandb_runner.summary["Best srcc_1"] = (
                max_plcc_2_a,
                max_srcc_2_a,
            )

        temp1 = plcc_2_a
        temp2 = srcc_2_a

        if epoch == config.TRAIN.EPOCHS - 1:
            # dataset B测试
            srcc_2_b, plcc_2_b, loss_2_b = validate(config, data_loader_val_B, t_model, epoch, 3,
                                                    len(dataset_val_B), writer, wandb_runner)

            if dist.get_rank() == 0:
                wandb_runner.log(
                    {"Validating srcc2": srcc_2_b, "Validating plcc2": plcc_2_b, "Epoch": epoch}
                )
            logger.info(
                f"PLCC and SRCC of the network on the {len(dataset_val_B)} test images: {plcc_2_b:.6f}, {srcc_2_b:.6f}"
            )
            if plcc_2_b >= max_plcc_2_b:
                max_plcc_2_b = max(max_plcc_2_b, plcc_2_b)
                max_srcc_2_b = srcc_2_b
            elif plcc_2_b < 0:
                max_srcc_2_b = 0
            logger.info(f"Max plcc_2: {max_plcc_2_b:.6f} Max srcc_2: {max_srcc_2_b:.6f}")
            if dist.get_rank() == 0:
                wandb_runner.summary["Best plcc_2"], wandb_runner.summary["Best srcc_2"] = (
                    max_plcc_2_b,
                    max_srcc_2_b,
                )
            temp3 = plcc_2_b
            temp4 = srcc_2_b

    # s_model = copy.deepcopy(t_model)
    # del t_model

    # 用来保存 A和B 的 examplar
    examplar_total = []

    # 取dataset A的examplar
    # for select exampar
    data_loader_train_ex1 = torch.utils.data.DataLoader(
        dataset_train_A,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
    )
    if len(dataset_train_A) > 20000:
        # for give label
        logger.info("get_random_examplar")
        partial_dataset_train_A = []
        partial_dataset_train_A = get_random_examplar(partial_dataset_train_A, dataset_train_A, len(dataset_train_A) // 20000)
        partial_data_loader_train_ex1 = torch.utils.data.DataLoader(
            partial_dataset_train_A,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
        )
    if dist.get_rank() == 0:
        logger.info("get_regressor")
    # 416
    regressor = KernelRidge(kernel='rbf', gamma=0.0001)
    regressor = get_regressor(t_model, regressor, partial_data_loader_train_ex1, len(partial_dataset_train_A),
                                     config.DATA.BATCH_SIZE, logger)

    if dist.get_rank() == 0:
        logger.info("get_A_examplar1")

    examplar_total = get_A_examplar1(examplar_total, dataset_train_A, data_loader_train_ex1, t_model, config.DATA.EXAMPLAR_STEP, logger)
    # examplar_total = get_best_examplar(config, examplar_total, dataset_train_A, t_model, config.DATA.EXAMPLAR_STEP, criterion, config.DATA.BATCH_SIZE)
    if dist.get_rank() == 0:
        logger.info(f"len(examplar_total):{len(examplar_total)}")

    # 保存B已经训练过的数据
    trained_index = [[]for i in range(config.DATA.SLICE_NUMBER)]

    for slice_num in range(config.DATA.SLICE_NUMBER):
        logger.info(f"slice{slice_num} training")
        train_index = []
        for i, index in enumerate(config.SET1.TRAIN_INDEX):
            if(i % config.DATA.SLICE_NUMBER == slice_num):
                train_index.append(index)
                # trained_index.append(index)
                trained_index[slice_num].append(index)

        # dataset B 分块 for train
        dataset_train1_teacher = build_IQA_dataset_krr_teacher(config, train_index)

        # pseudo_label = get_krr_label(config, t_model, partial_data_loader_train_ex1, len(partial_dataset_train_A),
        #                              dataset_train1_teacher, config.DATA.BATCH_SIZE1, logger)
        pseudo_label = get_krr_label(config, regressor, t_model, dataset_train1_teacher, config.DATA.BATCH_SIZE1, logger)
        dataset_train1_student = build_IQA_dataset_krr_student(config, train_index, pseudo_label, examplar_total)
        data_loader_train1, mixup_fn = IQA_build_loader_knn(config, dataset_train1_student)

        if dist.get_rank() == 0:
            logger.info(f"len(dataset_train1_teacher):{len(dataset_train1_teacher)}")
            logger.info(f"len(dataset_train1_student):{len(dataset_train1_student)}")
            logger.info(f"len(data_loader_train1):{len(data_loader_train1)}")
            
        # dataset B 分块 的optimizer和lr_scheduler
        optimizer_B = build_optimizer1(config, t_model)
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            lr_scheduler_B = build_scheduler3(
                config, optimizer_B, len(data_loader_train1) // config.TRAIN.ACCUMULATION_STEPS
            )
        else:
            lr_scheduler_B = build_scheduler3(config, optimizer_B, len(data_loader_train1))

        max_plcc_2_a = 0.0
        max_srcc_2_a = 0.0
        max_plcc_2_b = 0.0
        max_srcc_2_b = 0.0

        for epoch in range(config.TRAIN.START_EPOCH_STU1, config.TRAIN.EPOCHS_STU1):
            data_loader_train1.sampler.set_epoch(epoch)

            if dist.get_rank() == 0 and (
                    epoch % config.SAVE_FREQ == 1 or epoch == (config.TRAIN.EPOCHS_STU1 - 1)
            ):
                save_checkpoint1(
                    config,
                    epoch,
                    t_model_without_ddp,
                    slice_num,
                    max_plcc_2_a,
                    max_plcc_2_b,
                    optimizer_B,
                    lr_scheduler_B,
                    loss_scaler2,
                    logger,
                    save_num
                )

            train_one_epoch2(config, t_model, criterion, data_loader_train1, slice_num,
                            optimizer_B, epoch, mixup_fn, lr_scheduler_B, loss_scaler2, writer, wandb_runner)

            # dataset A测试
            srcc_2_a = 0.0
            plcc_2_a = 0.0
            if epoch == config.TRAIN.EPOCHS_STU1 - 1:

                # dataset A测试
                srcc_2_a, plcc_2_a, loss_2_a = validate2(config, data_loader_val_A, t_model, epoch, slice_num, 2,
                                                         len(dataset_val_A), writer, wandb_runner)
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

            # dataset B测试
            srcc_2_b, plcc_2_b, loss_2_b = validate2(config, data_loader_val_B, t_model, epoch, slice_num, 3,
                                                     len(dataset_val_B), writer, wandb_runner)

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

            if dist.get_rank() == 0:
                wandb_runner.log(
                    {f"slice{slice_num}Validating SRCC1": srcc_2_a,
                     f"slice{slice_num}Validating PLCC1": plcc_2_a,
                     f"slice{slice_num}Validating SRCC2": srcc_2_b,
                     f"slice{slice_num}Validating PLCC2": plcc_2_b,
                     f"slice{slice_num}Epoch": epoch}
                )
            logger.info(
                f"slice{slice_num} PLCC1 and SRCC1 of the network on the {len(dataset_val_A)} test images: {plcc_2_a:.6f}, {srcc_2_a:.6f}; "
            )
            logger.info(
                f"slice{slice_num}PLCC2 and SRCC2 of the network on the {len(dataset_val_B)} test images: {plcc_2_b:.6f}, {srcc_2_b:.6f}"
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

        if slice_num < config.DATA.SLICE_NUMBER-1:
            logger.info("get_A_examplar")
            # select examplar
            examplar_total = []
            # select data A examplar
            examplar_total = get_A_examplar1(examplar_total, dataset_train_A, data_loader_train_ex1, t_model, config.DATA.EXAMPLAR_STEP, logger)
            # examplar_total = get_best_examplar(config, examplar_total, dataset_train_A, t_model,
            #                                    config.DATA.EXAMPLAR_STEP, criterion, config.DATA.BATCH_SIZE)
            logger.info("get_B_examplar")
            for i in range(slice_num+1):
                # for select examplar
                dataset_train2_teacher = build_IQA_dataset_krr_teacher(config, trained_index[i])
                logger.info("get_krr_label")
                pseudo_label_examplar = get_krr_label(config, regressor, t_model, dataset_train2_teacher, config.DATA.BATCH_SIZE1, logger)
                dataset_train2_student = build_IQA_dataset_krr_student(config, trained_index[i], pseudo_label_examplar)
                data_loader_train_ex2 = torch.utils.data.DataLoader(
                    dataset_train2_student,
                    batch_size=config.DATA.BATCH_SIZE1,
                    num_workers=config.DATA.NUM_WORKERS,
                    pin_memory=config.DATA.PIN_MEMORY,
                )
                # select data B examplar
                examplar_total = get_A_examplar1(examplar_total, dataset_train2_student, data_loader_train_ex2, t_model, config.DATA.EXAMPLAR_STEP1, logger)
                # examplar_total = get_best_examplar(config, examplar_total, pseudo_label_data_examplar, t_model,
                #                                    config.DATA.EXAMPLAR_STEP1, criterion, config.DATA.BATCH_SIZE1)

            if dist.get_rank() == 0:
                logger.info(f"len(examplar_total):{len(examplar_total)}")


    writer.close()
    if dist.get_rank() == 0:
        wandb_runner.finish()
        logging.shutdown()
    else:
        logging.shutdown()
    dist.barrier()
    return temp1, temp2, temp3, temp4, plcc_2_a, srcc_2_a, plcc_2_b, srcc_2_b


def train_one_epoch2(
    config,
    s_model,
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
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)

        targets = targets.cuda(non_blocking=True)
        if mixup_fn is not None:
             samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            # outputs, aux_score = model(samples)
            outputs, feat = s_model(samples)

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
    train_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
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
    return


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
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            # outputs, aux_score = model(samples)
            outputs, feat = s_model(samples)
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
                f"Train: [{epoch+1}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
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
    train_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
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
    return


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
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            # output, _ = model(images)
            output, feat = s_model(images)
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

    for idx, (images, _, target, _) in enumerate(data_loader):
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

    plcc_srcc1 = []
    plcc_srcc2 = []
    plcc_srcc3 = []

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
            # print(sel_path1)
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
        # print("config.SET.TRAIN_INDEX:", config.SET.TRAIN_INDEX)
        config.SET.TEST_INDEX = sel_num[int(round(0.8 * len(sel_num))): len(sel_num)]
        # print("config.SET.TEST_INDEX:", config.SET.TEST_INDEX)

        config.SET1.TRAIN_INDEX = sel_num1[0: int(round(0.8 * len(sel_num1)))]
        # print("config.SET1.TRAIN_INDEX:", config.SET1.TRAIN_INDEX)
        config.SET1.TEST_INDEX = sel_num1[int(round(0.8 * len(sel_num1))): len(sel_num1)]
        # print("config.SET1.TEST_INDEX:", config.SET1.TEST_INDEX)
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

        # 计算平均性能以及提升
        plcc1_notrain, srcc1_notrain, plcc2_notrain, srcc2_notrain, plcc1_train, srcc1_train, plcc2_train, srcc2_train \
            = main(config, i)
        plcc_srcc_item = (plcc1_notrain, srcc1_notrain, plcc2_notrain, srcc2_notrain,
                          plcc1_train, srcc1_train, plcc2_train, srcc2_train)
        plcc_srcc1.append(plcc_srcc_item)
        plcc_srcc2.append(plcc_srcc_item)
        plcc_srcc3.append(plcc_srcc_item)

        # if i == num-1:
        if i % 2 == 1 and i > 0:
            plcc_srcc1.sort(key=lambda x: (x[4], x[5]), reverse=True)
            plcc_srcc2.sort(key=lambda x: (x[6], x[7]), reverse=True)
            plcc_srcc3.sort(key=lambda x: (x[6] - x[2], x[7] - x[3]), reverse=True)
            logger.info(f"plcc_srcc1:{plcc_srcc1}")
            logger.info(f"plcc_srcc2:{plcc_srcc2}")
            logger.info(f"plcc_srcc3:{plcc_srcc3}")

            plcc1_notrain_total = 0.0
            srcc1_notrain_total = 0.0
            plcc2_notrain_total = 0.0
            srcc2_notrain_total = 0.0
            plcc1_train_total = 0.0
            srcc1_train_total = 0.0
            plcc2_train_total = 0.0
            srcc2_train_total = 0.0

            for plcc1_notrain, srcc1_notrain, plcc2_notrain, srcc2_notrain, \
                plcc1_train, srcc1_train, plcc2_train, srcc2_train in plcc_srcc1[0:(i + 1) // 2]:
                plcc1_notrain_total += plcc1_notrain
                srcc1_notrain_total += srcc1_notrain
                plcc2_notrain_total += plcc2_notrain
                srcc2_notrain_total += srcc2_notrain
                plcc1_train_total += plcc1_train
                srcc1_train_total += srcc1_train
                plcc2_train_total += plcc2_train
                srcc2_train_total += srcc2_train

            plcc1_notrain_average = '%.4f' % (plcc1_notrain_total / ((i + 1) // 2))
            srcc1_notrain_average = '%.4f' % (srcc1_notrain_total / ((i + 1) // 2))
            plcc2_notrain_average = '%.4f' % (plcc2_notrain_total / ((i + 1) // 2))
            srcc2_notrain_average = '%.4f' % (srcc2_notrain_total / ((i + 1) // 2))
            plcc1_train_average = '%.4f' % (plcc1_train_total / ((i + 1) // 2))
            srcc1_train_average = '%.4f' % (srcc1_train_total / ((i + 1) // 2))
            plcc2_train_average = '%.4f' % (plcc2_train_total / ((i + 1) // 2))
            srcc2_train_average = '%.4f' % (srcc2_train_total / ((i + 1) // 2))
            logger.info(f"plcc1_notrain, plcc1_train:{plcc1_notrain_average},{plcc1_train_average};"
                        f"srcc1_notrain, srcc1_train:{srcc1_notrain_average},{srcc1_train_average};"
                        f"plcc2_notrain, plcc2_train:{plcc2_notrain_average},{plcc2_train_average};"
                        f"srcc2_notrain, srcc2_train:{srcc2_notrain_average},{srcc2_train_average}")

            plcc1_notrain_total = 0.0
            srcc1_notrain_total = 0.0
            plcc2_notrain_total = 0.0
            srcc2_notrain_total = 0.0
            plcc1_train_total = 0.0
            srcc1_train_total = 0.0
            plcc2_train_total = 0.0
            srcc2_train_total = 0.0
            for plcc1_notrain, srcc1_notrain, plcc2_notrain, srcc2_notrain, \
                plcc1_train, srcc1_train, plcc2_train, srcc2_train in plcc_srcc2[0:(i + 1) // 2]:
                plcc1_notrain_total += plcc1_notrain
                srcc1_notrain_total += srcc1_notrain
                plcc2_notrain_total += plcc2_notrain
                srcc2_notrain_total += srcc2_notrain
                plcc1_train_total += plcc1_train
                srcc1_train_total += srcc1_train
                plcc2_train_total += plcc2_train
                srcc2_train_total += srcc2_train

            plcc1_notrain_average = '%.4f' % (plcc1_notrain_total / ((i + 1) // 2))
            srcc1_notrain_average = '%.4f' % (srcc1_notrain_total / ((i + 1) // 2))
            plcc2_notrain_average = '%.4f' % (plcc2_notrain_total / ((i + 1) // 2))
            srcc2_notrain_average = '%.4f' % (srcc2_notrain_total / ((i + 1) // 2))
            plcc1_train_average = '%.4f' % (plcc1_train_total / ((i + 1) // 2))
            srcc1_train_average = '%.4f' % (srcc1_train_total / ((i + 1) // 2))
            plcc2_train_average = '%.4f' % (plcc2_train_total / ((i + 1) // 2))
            srcc2_train_average = '%.4f' % (srcc2_train_total / ((i + 1) // 2))
            logger.info(f"plcc1_notrain, plcc1_train:{plcc1_notrain_average},{plcc1_train_average};"
                        f"srcc1_notrain, srcc1_train:{srcc1_notrain_average},{srcc1_train_average};"
                        f"plcc2_notrain, plcc2_train:{plcc2_notrain_average},{plcc2_train_average};"
                        f"srcc2_notrain, srcc2_train:{srcc2_notrain_average},{srcc2_train_average}")

            plcc1_notrain_total = 0.0
            srcc1_notrain_total = 0.0
            plcc2_notrain_total = 0.0
            srcc2_notrain_total = 0.0
            plcc1_train_total = 0.0
            srcc1_train_total = 0.0
            plcc2_train_total = 0.0
            srcc2_train_total = 0.0
            for plcc1_notrain, srcc1_notrain, plcc2_notrain, srcc2_notrain, \
                plcc1_train, srcc1_train, plcc2_train, srcc2_train in plcc_srcc3[0:(i + 1) // 2]:
                plcc1_notrain_total += plcc1_notrain
                srcc1_notrain_total += srcc1_notrain
                plcc2_notrain_total += plcc2_notrain
                srcc2_notrain_total += srcc2_notrain
                plcc1_train_total += plcc1_train
                srcc1_train_total += srcc1_train
                plcc2_train_total += plcc2_train
                srcc2_train_total += srcc2_train

            plcc1_notrain_average = '%.4f' % (plcc1_notrain_total / ((i + 1) // 2))
            srcc1_notrain_average = '%.4f' % (srcc1_notrain_total / ((i + 1) // 2))
            plcc2_notrain_average = '%.4f' % (plcc2_notrain_total / ((i + 1) // 2))
            srcc2_notrain_average = '%.4f' % (srcc2_notrain_total / ((i + 1) // 2))
            plcc1_train_average = '%.4f' % (plcc1_train_total / ((i + 1) // 2))
            srcc1_train_average = '%.4f' % (srcc1_train_total / ((i + 1) // 2))
            plcc2_train_average = '%.4f' % (plcc2_train_total / ((i + 1) // 2))
            srcc2_train_average = '%.4f' % (srcc2_train_total / ((i + 1) // 2))
            logger.info(f"plcc1_notrain, plcc1_train:{plcc1_notrain_average},{plcc1_train_average};"
                        f"srcc1_notrain, srcc1_train:{srcc1_notrain_average},{srcc1_train_average};"
                        f"plcc2_notrain, plcc2_train:{plcc2_notrain_average},{plcc2_train_average};"
                        f"srcc2_notrain, srcc2_train:{srcc2_notrain_average},{srcc2_train_average}")
        logger.handlers.clear()
