from torch import tensor
from torchvision import transforms
import json
import numpy
from PIL import Image
import os
import torch
from torch.utils.data import DataLoader,TensorDataset

class LoadTrainConfig():

    def __init__(self,configure_filePath,resize=None,center_crop=None):

        IMAGENET_MEAN = tensor([.485, .456, .406])
        IMAGENET_STD = tensor([.229, .224, .225])
        transfoms_paras = [
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                    ]
        if resize!=None:
            transfoms_paras.append(transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC))
        if center_crop!=None:
            transfoms_paras.append(transforms.CenterCrop(center_crop))
        
        self.loader=transforms.Compose(transfoms_paras)

        with open(configure_filePath) as json_file:
            self.data = json.load(json_file)
        
    def genTrainDS(self,train_imgs_folder):
        train_ims = []
        train_labels = []

        for img_id in self.data['train_ids']:
            img_path = os.path.join(train_imgs_folder, img_id)
            
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


if __name__ == "__main__":
    configPath = 'config/semi/percentage_1.json'
    loading_train = LoadTrainConfig(configPath)
    train_ds = loading_train.genTrainDS('datasets/THz_Body/train/good')