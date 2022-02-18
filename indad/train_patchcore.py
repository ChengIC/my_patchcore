from models import PatchCore
from save_utils import saveModelPath
import numpy 
import torch
import warnings 
from torch import tensor
from torchvision import transforms
import json
import numpy
from PIL import Image
import os
from torch.utils.data import DataLoader,TensorDataset

warnings.filterwarnings("ignore")


class train_patchcore():
    def __init__(self,configPath,train_imgs_folder,
                resize=None,center_crop=None,scaling_factor=None,
                f_coreset=.20,backbone_name="wide_resnet50_2",TimeStamp=None):
        
        self.configPath=configPath
        self.train_imgs_folder=train_imgs_folder
        self.resize=resize
        self.center_crop=center_crop
        self.scaling_factor=scaling_factor
        self.f_coreset=f_coreset
        self.backbone_name=backbone_name
        self.TimeStamp=TimeStamp
        with open(configPath) as json_file:
            self.data = json.load(json_file)

        self.model=PatchCore(
                    f_coreset=f_coreset, 
                    backbone_name=backbone_name,
                )

        self.train_tar,self.train_path,self.model_path=saveModelPath(self.configPath,self.TimeStamp)

        IMAGENET_MEAN = tensor([.485, .456, .406])
        IMAGENET_STD = tensor([.229, .224, .225])
        transfoms_paras = [
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                    ]
        if resize!=None:
            transfoms_paras.append(transforms.Resize(self.resize, interpolation=transforms.InterpolationMode.BICUBIC))
        if center_crop!=None:
            transfoms_paras.append(transforms.CenterCrop(center_crop))
        if scaling_factor!=None:
            width = int(self.data['original_imgsz'][0]*scaling_factor)
            height = int(self.data['original_imgsz'][1]*scaling_factor)
            self.resize=[width,height]
            transfoms_paras.append(transforms.Resize(self.resize, interpolation=transforms.InterpolationMode.BICUBIC))

        self.loader=transforms.Compose(transfoms_paras)

    def genTrainDS(self):
        train_ims = []
        train_labels = []

        for img_id in self.data['train_ids']:
            img_path = os.path.join(self.train_imgs_folder, img_id)
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

    def saveTrainConfig(self):

        self.data['imgsz'] = self.resize
        self.data['center_crop'] = self.center_crop
        self.data['scaling_factor'] = self.scaling_factor
        self.data['train_imgs_folder'] = self.train_imgs_folder
        self.data['backbone_name'] = self.backbone_name
        self.data['TimeStamp'] = self.TimeStamp

        json_string = json.dumps(self.data)
        json_filePath = os.path.join(self.model_path,'training_config.json')
        with open(json_filePath, 'w') as outfile:
            outfile.write(json_string)
    
    def run(self):
        train_ds = self.genTrainDS()
        tobesaved = self.model.fit(train_ds)
        torch.save(tobesaved, self.train_tar)
        torch.save(self.model.state_dict(), self.train_path)
        self.saveTrainConfig()