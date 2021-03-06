
import numpy as np
import torch
from torch import tensor
import os
from torch.utils.data import DataLoader,TensorDataset
from PIL import Image
from torchvision import transforms
import random
import cv2

random.seed(20220308)

IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']  # include image suffixes

class Process_Train_Folder:

    def check_size_folder(training_folder):
        h_dict=[]
        w_dict=[]
        for img_file in os.listdir(training_folder):
            if img_file.split('.')[-1] in IMG_FORMATS:
                img_path = os.path.join(training_folder,img_file)
                im = cv2.imread(img_path)
                h, w, _ = im.shape
                if len(h_dict)==0:
                    h_dict.append(h)
                else:
                    if h_dict[0]!=h:
                        return False

                if len(w_dict)==0:
                    w_dict.append(w)
                else:
                    if w_dict[0]!=w:
                        return False
        return True


    def mean_size_folder(training_folder):
        height_list=[]
        width_list=[]
        for img_file in os.listdir(training_folder):
            if img_file.split('.')[-1] in IMG_FORMATS:
                img_path = os.path.join(training_folder,img_file)
                im = cv2.imread(img_path)
                h, w, _ = im.shape
                height_list.append(h)
                width_list.append(w)
        width_list = np.array(width_list)
        height_list = np.array(height_list)
        return [int(np.mean(height_list)),int(np.mean(width_list))]

class genDS():
    def __init__(self,
                training_folder,
                scale=1.0,
                resize_box=None):
        self.resize_box = resize_box
        
        IMAGENET_MEAN = tensor([.485, .456, .406])
        IMAGENET_STD = tensor([.229, .224, .225])
        if resize_box==None:
            if scale==1:
                self.resize_box=Process_Train_Folder.mean_size_folder(training_folder)
                transfoms_paras = [
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]

            else:
                self.resize_box=Process_Train_Folder.mean_size_folder(training_folder)
                self.resize_box = [int(self.resize_box[0]*scale),int(self.resize_box[1]*scale)]
                # print (self.resize_box)
                transfoms_paras = [
                            transforms.Resize(self.resize_box, interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                    ]
        else:
            transfoms_paras = [
                            transforms.Resize(self.resize_box, interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                    ]
        self.training_folder=training_folder
        self.loader=transforms.Compose(transfoms_paras)
    
    def genTrainDS(self,percentage=1):
        ds_info = {}
        train_ims = []
        train_labels = []
        len_folder = len(os.listdir(self.training_folder))
        folder_files_list = os.listdir(self.training_folder)
        random.shuffle(folder_files_list)
        training_list = folder_files_list[0:int(len_folder*percentage)]

        ds_info['im_id'] = []
        for img_id in training_list:
            if img_id.split('.')[-1] in IMG_FORMATS:
                img_path = os.path.join(self.training_folder, img_id)
                train_im = Image.open(img_path).convert('RGB')
                width, height = train_im.size
                train_im = self.loader(train_im)
                train_label = tensor([0])
                train_ims.append(train_im.numpy())
                train_labels.append(train_label.numpy())

                ds_info['im_id'].append(img_id)

        train_ims = np.array(train_ims)
        train_labels = np.array(train_labels)
        print ('Training Tensor Shape is' + str(train_ims.shape))
        train_ims = torch.from_numpy(train_ims)
        train_labels = torch.from_numpy(train_labels)
        train_data = TensorDataset(train_ims,train_labels)
        train_ds = DataLoader(train_data)

        ds_info['box'] = self.resize_box

        return train_ds, self.loader, ds_info

    def genTrainDS_by_config(self,config):
        ds_info = {}
        train_ims = []
        train_labels = []

        ds_info['im_id'] = []
        for img_id in config['img_ids']:
            if img_id.split('.')[-1] in IMG_FORMATS:
                img_path = os.path.join(self.training_folder, img_id)
                train_im = Image.open(img_path).convert('RGB')
                train_im = self.loader(train_im)
                train_label = tensor([0])
                train_ims.append(train_im.numpy())
                train_labels.append(train_label.numpy())

                ds_info['im_id'].append(img_id)

        train_ims = np.array(train_ims)
        train_labels = np.array(train_labels)
        print ('Training Tensor Shape is' + str(train_ims.shape))
        train_ims = torch.from_numpy(train_ims)
        train_labels = torch.from_numpy(train_labels)
        train_data = TensorDataset(train_ims,train_labels)
        train_ds = DataLoader(train_data)

        ds_info['box'] = self.resize_box

        return train_ds, self.loader, ds_info