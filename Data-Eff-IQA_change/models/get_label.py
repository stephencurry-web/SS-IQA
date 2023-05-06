import torch
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import torch.distributed as dist
from .KneighborsRegressor import KneighborsRegressor
from .knn_find import knn_find
from .KernelRegression import KernelRegression
from .LocallyWeightedRegression import LocallyWeightedRegression

# def get_label(model, data_loader_train_ex1, labeled_len, unlabeled_data, batch_size, logger):
#     data_loader_train_ex2 = torch.utils.data.DataLoader(
#         unlabeled_data,
#         batch_size=batch_size
#     )
#     unlabeled_len = len(unlabeled_data)
#     if dist.get_rank() == 0:
#         logger.info(f"unlabeled_len:{unlabeled_len}")
#     labeled_features = np.zeros((labeled_len, 192))
#     labeled_labels = np.zeros((labeled_len, 1))
#     unlabeled_features = np.zeros((unlabeled_len, 192))
#     jia_labels = torch.zeros(unlabeled_len, 1)
#     for idx, (image, _, targets) in enumerate(data_loader_train_ex1):
#         with torch.no_grad():
#             image = image.cuda(non_blocking=True)
#             # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
#             B = image.size()[0]
#             _, feat = model(image)
#             # jia_labels.append(jia_label)
#             feat = feat.view(B, -1)
#             feat = feat.cpu().numpy()
#             target = targets.detach().cpu().numpy()
#             for i, _ in enumerate(feat):
#                 labeled_features[(idx * batch_size + i):(idx * batch_size + i + 1), :] = feat[i:i+1, :]
#             for i, _ in enumerate(target):
#                 labeled_labels[(idx * batch_size + i):(idx * batch_size + i + 1), :] = target[i:i+1]
#
#
#     for idx, (image, image1, targets) in enumerate(data_loader_train_ex2):
#         with torch.no_grad():
#             image1 = image1.cuda(non_blocking=True)
#             # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
#             B = image1.size()[0]
#             jia_label, feat = model(image1)
#             for i, _ in enumerate(jia_label):
#                 jia_labels[idx * batch_size + i:idx * batch_size + i + 1, :] = jia_label[i]
#             feat = feat.view(B, -1)
#             feat = feat.cpu().numpy()
#             for i, _ in enumerate(feat):
#                 unlabeled_features[(idx * batch_size + i):(idx * batch_size + i + 1), :] = feat[i:i+1, :]
#     # unlabeled_features = np.array(unlabeled_features)
#     # 使用knn得到伪标签
#     # all_label = knn_find(labeled_features, labeled_labels, unlabeled_features)
#     # 使用核回归得到伪标签
#     # regressor = KernelRegression(kernel='gaussian', bandwidth=0.5)
#     # # regressor = LocallyWeightedRegression(bandwidth=0.5)
#     # regressor.fit(labeled_features, labeled_labels)
#
#
#     # 核岭回归
#     # regressor = KernelRidge(kernel='rbf', gamma=0.005208)
#     # regressor.fit(labeled_features, labeled_labels)
#     # all_label = regressor.predict(unlabeled_features)
#     # knn算法
#     regressor = KneighborsRegressor(5)
#     regressor.fit(labeled_features, labeled_labels)
#     all_label = np.array([regressor.predict(x) for x in unlabeled_features])
#
#     if dist.get_rank() == 0:
#         logger.info(f"all_label.shape:{all_label.shape}")
#     max_val = np.amax(all_label)
#     min_val = np.amin(all_label)
#     if dist.get_rank() == 0:
#         logger.info(f"最大值:{max_val}")
#         logger.info(f"最小值:{min_val}")
#     all_label = torch.from_numpy(all_label)
#     if dist.get_rank() == 0:
#         logger.info(f"all_label:{all_label}")
#
#     unlabeled_res = []
#     count1 = 0.0
#     count2 = 0.0
#     for i, (image, image1, label) in enumerate(unlabeled_data):
#         count1 += abs(label - all_label[i].item())
#         count2 += abs(label - jia_labels[i].item())
#         unlabeled_res.append((image, image1, all_label[i].item()))
#     if dist.get_rank() == 0:
#         logger.info(f"count1:{count1 / unlabeled_len}")
#         logger.info(f"count2:{count2 / unlabeled_len}")
#     return unlabeled_res


def get_pseudo_label(config, model, unlabeled_data, batch_size, logger):
    data_loader_train_ex2 = torch.utils.data.DataLoader(
        unlabeled_data,
        batch_size=batch_size,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
    )
    unlabeled_len = len(unlabeled_data)
    if dist.get_rank() == 0:
        logger.info(f"unlabeled_len:{unlabeled_len}")

    jia_labels = torch.zeros(unlabeled_len, 1)

    for idx, image1 in enumerate(data_loader_train_ex2):
        with torch.no_grad():
            image1 = image1.cuda(non_blocking=True)
            # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            jia_label, _ = model(image1)
            for i, _ in enumerate(jia_label):
                jia_labels[idx * batch_size + i:idx * batch_size + i + 1, :] = jia_label[i]
    logger.info("get jia_label")
    # unlabeled_res = []
    # for i, (image1, label) in enumerate(unlabeled_data):
    #     unlabeled_res.append((image1, jia_labels[i].item()))
    # return unlabeled_res
    return jia_labels