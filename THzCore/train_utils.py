
import numpy
import torch
from torch import tensor
import os
from torch.utils.data import DataLoader,TensorDataset
from PIL import Image
from torchvision import transforms
import random

class genDS():

    def __init__(self,region_type=None):
        self.region_type=region_type
        if self.region_type == 'main_body':
            resize_box = [136,280]
        elif self.region_type == 'arm':
            resize_box = [100,182]
        elif self.region_type == 'leg':
            resize_box = [85,300]
        else:
            raise Exception("undefined region types")

        IMAGENET_MEAN = tensor([.485, .456, .406])
        IMAGENET_STD = tensor([.229, .224, .225])
        transfoms_paras = [
                        transforms.Resize(resize_box, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
        self.loader=transforms.Compose(transfoms_paras)

    def genTrainDS(self,training_folder,percentage=1):
        train_ims = []
        train_labels = []
        len_folder = len(os.listdir(training_folder))
        folder_files_list = os.listdir(training_folder)
        random.shuffle(folder_files_list)
        training_list = folder_files_list[0:int(len_folder*percentage)]
        for img_id in training_list:
            if '.jpg' in img_id:
                img_path = os.path.join(training_folder, img_id)
                train_im = Image.open(img_path).convert('RGB')
                train_im = self.loader(train_im)

                train_label = tensor([0])
                train_ims.append(train_im.numpy())
                train_labels.append(train_label.numpy())

        train_ims = numpy.array(train_ims)
        train_labels = numpy.array(train_labels)
        print ('Training Tensor Shape is' + str(train_ims.shape))

        train_ims = torch.from_numpy(train_ims)
        train_labels = torch.from_numpy(train_labels)
        train_data = TensorDataset(train_ims,train_labels)
        train_ds = DataLoader(train_data)

        return train_ds