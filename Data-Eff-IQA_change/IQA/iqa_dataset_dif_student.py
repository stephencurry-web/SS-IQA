import csv
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image
from scipy import io


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename

def transfer(database, label):
    if database == 'live':
        label = 100 - label
    elif database == "csiq":
        label = 100 - label * 100.0
    elif database == "tid2013":
        label = label / 9 * 100.0
    elif database == "kadid":
        label = (label - 1) * 25.0
    return label


class KONIQDATASET7(data.Dataset):
    def __init__(self, root, index, patch_num, examplar, transform=None,  pseudo_label=None):
        super(KONIQDATASET7, self).__init__()

        self.data_path = root
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, "koniq10k_scores_and_distributions.csv")
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row["image_name"])
                mos = np.array(float(row["MOS_zscore"])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        count = 0
        for i, item in enumerate(index):
            for j in range(patch_num):
                path = os.path.join(root, "1024x768", imgname[item])
                # image = self._load_image(path)
                # if transform is not None:
                #     image = transform(image)
                sample.append(
                    (path,
                     pseudo_label[count].item()
                     )
                )
                count += 1
        self.count = count
        if examplar is not None:
            for _, item in enumerate(examplar):
                sample.append(item)

        self.samples = sample
        self.transform = transform
        

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        # path1, target = self.samples[index]
        # sample1 = self._load_image(path1)
        #
        # if self.transform is not None:
        #     sample1 = self.transform(sample1)
        # if self.pseudo_label is not None:
        #     target = self.pseudo_label[index].item()
        if index < self.count:
            path, target = self.samples[index]
            image = self._load_image(path)
            if self.transform is not None:
                image = self.transform(image)
        else:
            image, target = self.samples[index]
        return image, target

    def __len__(self):
        length = len(self.samples)
        return length


class LIVECDATASET7(data.Dataset):
    def __init__(self, root, index, patch_num, examplar, transform=None,  pseudo_label=None):

        imgpath = io.loadmat(os.path.join(root, "Data", "AllImages_release.mat"))
        imgpath = imgpath["AllImages_release"]
        imgpath = imgpath[7:1169]
        mos = io.loadmat(os.path.join(root, "Data", "AllMOS_release.mat"))
        labels = mos["AllMOS_release"].astype(np.float32)
        labels = labels[0][7:1169]

        sample = []
        count = 0
        for i, item in enumerate(index):
            for aug in range(patch_num):
                # sample.append(
                #     (os.path.join(root, "Images", imgpath[item][0][0]),
                #      labels[item])
                # )
                path = os.path.join(root, "Images", imgpath[item][0][0])
                # image = self._load_image(path)
                # if transform is not None:
                #     image = transform(image)
                sample.append(
                    (path,
                     pseudo_label[count].item()
                     )
                )
                count += 1
        self.count = count
        if examplar is not None:
            for _, item in enumerate(examplar):
                sample.append(item)

        self.samples = sample
        self.transform = transform
        

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # path1, target = self.samples[index]
        # sample1 = self._load_image(path1)
        #
        # if self.transform is not None:
        #     sample1 = self.transform(sample1)
        # if self.pseudo_label is not None:
        #     target = self.pseudo_label[index].item()

        if index < self.count:
            path, target = self.samples[index]
            image = self._load_image(path)
            if self.transform is not None:
                image = self.transform(image)
        else:
            image, target = self.samples[index]
        return image, target


    def __len__(self):
        length = len(self.samples)
        return length


class LIVEDataset7(data.Dataset):
    def __init__(self, root, index, patch_num, examplar, transform=None,  pseudo_label=None):

        refpath = os.path.join(root, "refimgs")
        refname = getFileName(refpath, ".bmp")

        jp2kroot = os.path.join(root, "jp2k")
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(root, "jpeg")
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(root, "wn")
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(root, "gblur")
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(root, "fastfading")
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        dmos = io.loadmat(os.path.join(root, "dmos_realigned.mat"))
        labels = dmos["dmos_new"].astype(np.float32)

        orgs = dmos["orgs"]
        refnames_all = io.loadmat(os.path.join(root, "refnames_all.mat"))
        refnames_all = refnames_all["refnames_all"]

        refname.sort()
        sample = []
        count = 0

        for i in range(0, len(index)):
            train_sel = refname[index[i]] == refnames_all
            train_sel = train_sel * ~orgs.astype(np.bool_)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[1].tolist()

            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    # sample.append((imgpath[item], transfer("live", labels[0][item])))

                    path = os.path.join(root, "1024x768", imgpath[item])
                    # image = self._load_image(path)
                    # if transform is not None:
                    #     image = transform(image)
                    sample.append(
                        (path,
                         pseudo_label[count].item()
                         )
                    )
                    count += 1
        self.count = count
        if examplar is not None:
            for _, item in enumerate(examplar):
                sample.append(item)

        self.samples = sample
        self.transform = transform
        

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # path1, target = self.samples[index]
        # sample1 = self._load_image(path1)
        #
        # if self.transform is not None:
        #     sample1 = self.transform(sample1)
        # if self.pseudo_label is not None:
        #     target = self.pseudo_label[index].item()
        #
        # return sample1, target
        if index < self.count:
            path, target = self.samples[index]
            image = self._load_image(path)
            if self.transform is not None:
                image = self.transform(image)
        else:
            image, target = self.samples[index]
        return image, target

    def __len__(self):
        length = len(self.samples)
        return length

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = "%s%s%s" % ("img", str(index), ".bmp")
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename


class TID2013Dataset7(data.Dataset):

    def __init__(self, root, index, patch_num, examplar, transform=None,  pseudo_label=None):
        refpath = os.path.join(root, "reference_images")
        refname = getTIDFileName(refpath, ".bmp.BMP")
        txtpath = os.path.join(root, "mos_with_names.txt")
        fh = open(txtpath, "r")
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split("\n")
            words = line[0].split()
            if "_01" in words[1] or "_08" in words[1] or "_10" in words[1] or "_11" in words[1]:
                imgnames.append((words[1]))
                target.append(words[0])
                ref_temp = words[1].split("_")
                refnames_all.append(ref_temp[0][1:])
        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        refname.sort()
        sample = []
        count = 0
        for i, item in enumerate(index):

            train_sel = refname[index[i]] == refnames_all
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    # sample.append(
                    #     (
                    #         os.path.join(root, "distorted_images", imgnames[item]),
                    #         transfer("tid2013", labels[item])
                    #     )
                    # )
                    path = os.path.join(root, "distorted_images", imgnames[item])
                    # image = self._load_image(path)
                    # if transform is not None:
                    #     image = transform(image)
                    sample.append(
                        (path,
                         pseudo_label[count].item()
                         )
                    )
                    count += 1

        self.count = count
        if examplar is not None:
            for _, item in enumerate(examplar):
                sample.append(item)

        self.samples = sample
        self.transform = transform
        

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        # path1, target = self.samples[index]
        # sample1 = self._load_image(path1)
        #
        # if self.transform is not None:
        #     sample1 = self.transform(sample1)
        # if self.pseudo_label is not None:
        #     target = self.pseudo_label[index].item()
        #
        # return sample1, target

        if index < self.count:
            path, target = self.samples[index]
            image = self._load_image(path)
            if self.transform is not None:
                image = self.transform(image)
        else:
            image, target = self.samples[index]
        return image, target

    def __len__(self):
        length = len(self.samples)
        return length


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


class CSIQDataset7(data.Dataset):
    def __init__(self, root, index, patch_num, examplar, transform=None,  pseudo_label=None):

        refpath = os.path.join(root, "src_imgs")
        refname = getFileName(refpath, ".png")
        txtpath = os.path.join(root, "csiq_label.txt")
        fh = open(txtpath, "r")
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split("\n")
            words = line[0].split()
            if "fnoise" in words[0] or "contrast" in words[0]:
                continue
            imgnames.append((words[0]))
            target.append(words[1])
            ref_temp = words[0].split(".")
            refnames_all.append(ref_temp[0] + "." + "png")

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []
        count = 0

        for i, item in enumerate(index):
            train_sel = refname[index[i]] == refnames_all
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    # sample.append(
                    #     (
                    #         os.path.join(root, "dst_imgs_all", imgnames[item]+".png"),
                    #         transfer("csiq", labels[item])
                    #     )
                    # )
                    path = os.path.join(root, "dst_imgs_all", imgnames[item]+".png")
                    # image = self._load_image(path)
                    # if transform is not None:
                    #     image = transform(image)
                    sample.append(
                        (path,
                         pseudo_label[count].item()
                         )
                    )
                    count += 1

        self.count = count
        if examplar is not None:
            for _, item in enumerate(examplar):
                sample.append(item)

        self.samples = sample
        self.transform = transform
        

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # path1, target = self.samples[index]
        # sample1 = self._load_image(path1)
        #
        # if self.transform is not None:
        #     sample1 = self.transform(sample1)
        # if self.pseudo_label is not None:
        #     target = self.pseudo_label[index].item()
        #
        # return sample1, target
        if index < self.count:
            path, target = self.samples[index]
            image = self._load_image(path)
            if self.transform is not None:
                image = self.transform(image)
        else:
            image, target = self.samples[index]
        return image, target

    def __len__(self):
        length = len(self.samples)
        return length


class KADIDDataset7(data.Dataset):
    def __init__(self, root, index, patch_num, examplar, transform=None,  pseudo_label=None):
        refpath = os.path.join(root, "reference_images")
        refname = getTIDFileName(refpath, ".png.PNG")

        imgnames = []
        target = []
        refnames_all = []

        csv_file = os.path.join(root, "dmos.csv")
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "_01_" in row["dist_img"] or "_09_" in row["dist_img"] or "_10_" in row["dist_img"] or "_11_" in row["dist_img"]:
                    imgnames.append(row["dist_img"])
                    refnames_all.append(row["ref_img"][1:3])
                    mos = np.array(float(row["dmos"])).astype(np.float32)
                    target.append(mos)
        # print(imgnames)
        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        refname.sort()
        sample = []
        count = 0
        for i, item in enumerate(index):
            train_sel = refname[index[i]] == refnames_all
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for _ in range(patch_num):
                    # sample.append(
                    #     (
                    #         os.path.join(root, "images", imgnames[item]),
                    #         transfer("kadid", labels[item])
                    #     )
                    # )
                    path = os.path.join(root, "images", imgnames[item])
                    # image = self._load_image(path)
                    # if transform is not None:
                    #     image = transform(image)
                    sample.append(
                        (path,
                         pseudo_label[count].item()
                         )
                    )
                    count += 1

        self.count = count
        if examplar is not None:
            for _, item in enumerate(examplar):
                sample.append(item)

        self.samples = sample
        self.transform = transform
        

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # path1, target = self.samples[index]
        # sample1 = self._load_image(path1)
        #
        # if self.transform is not None:
        #     sample1 = self.transform(sample1)
        # if self.pseudo_label is not None:
        #     target = self.pseudo_label[index].item()
        #
        # return sample1, target
        if index < self.count:
            path, target = self.samples[index]
            image = self._load_image(path)
            if self.transform is not None:
                image = self.transform(image)
        else:
            image, target = self.samples[index]
        return image, target

    def __len__(self):
        length = len(self.samples)
        return length


class SPAQDATASET7(data.Dataset):
    def __init__(self, root, index, patch_num, examplar, transform=None,  pseudo_label=None):
        super(SPAQDATASET7, self).__init__()

        self.data_path = root
        anno_folder = os.path.join(self.data_path, "Annotations")
        xlsx_file = os.path.join(anno_folder, "MOS and Image attribute scores.xlsx")
        read = pd.read_excel(xlsx_file)
        imgname = read["Image name"].values.tolist()
        mos_all = read["MOS"].values.tolist()
        for i in range(len(mos_all)):
            mos_all[i] = np.array(mos_all[i]).astype(np.float32)
        sample = []
        count = 0
        for _, item in enumerate(index):
            for _ in range(patch_num):
                # sample.append(
                #     (
                #         os.path.join(
                #             self.data_path,
                #             "TestImage",
                #             imgname[item],
                #         ),
                #         mos_all[item],
                #
                #     )
                # )
                path = os.path.join(
                            self.data_path,
                            "TestImage",
                            imgname[item],
                        )
                # image = self._load_image(path)
                # if transform is not None:
                #     image = transform(image)
                sample.append(
                    (path,
                     pseudo_label[count].item()
                     )
                )
                count += 1
        self.count = count
        if examplar is not None:
            for _, item in enumerate(examplar):
                sample.append(item)

        self.samples = sample
        self.transform = transform


    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # path1, target = self.samples[index]
        # sample1 = self._load_image(path1)
        #
        # if self.transform is not None:
        #     sample1 = self.transform(sample1)
        # if self.pseudo_label is not None:
        #     target = self.pseudo_label[index].item()
        #
        # return sample1, target
        if index < self.count:
            path, target = self.samples[index]
            image = self._load_image(path)
            if self.transform is not None:
                image = self.transform(image)
        else:
            image, target = self.samples[index]
        return image, target

    def __len__(self):
        length = len(self.samples)
        return length


class FBLIVEFolder7(data.Dataset):
    def __init__(self, root, index, patch_num, examplar, transform=None,  pseudo_label=None):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, "labels_image.csv")
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row["name"])
                mos = np.array(float(row["mos"])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        count = 0
        for i, item in enumerate(index):
            for aug in range(patch_num):
                # sample.append(
                #     (os.path.join(root, "database", imgname[item]),
                #      mos_all[item])
                # )
                path = os.path.join(root, "database", imgname[item])
                # image = self._load_image(path)
                # if transform is not None:
                #     image = transform(image)
                sample.append(
                    (path,
                     pseudo_label[count].item()
                     )
                )
                count += 1
        self.count = count
        if examplar is not None:
            for _, item in enumerate(examplar):
                sample.append(item)

        self.samples = sample
        self.transform = transform


    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # path1, target = self.samples[index]
        # sample1 = self._load_image(path1)
        #
        # if self.transform is not None:
        #     sample1 = self.transform(sample1)
        # if self.pseudo_label is not None:
        #     target = self.pseudo_label[index].item()
        #
        # return sample1, target
        if index < self.count:
            path, target = self.samples[index]
            image = self._load_image(path)
            if self.transform is not None:
                image = self.transform(image)
        else:
            image, target = self.samples[index]
        return image, target

    def __len__(self):
        length = len(self.samples)
        return length