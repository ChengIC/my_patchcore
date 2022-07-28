
from my_utils.ran_utils import *
import os 
import json
from models import PatchCore
from tqdm import tqdm
import numpy as np
import torch
from torch import tensor
import os
from torch.utils.data import DataLoader,TensorDataset
from PIL import Image
from torchvision import transforms


class TrainPatchCore ():
    def __init__(self, config_dir, model_dir):
        self.config_dir = config_dir
        self.model_dir = model_dir

    def genDataSet(self,config_data):

        # create dataloader
        IMAGENET_MEAN = tensor([.485, .456, .406])
        IMAGENET_STD = tensor([.229, .224, .225])
        transfoms_paras = [
                            transforms.ToTensor(),
                            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                        ]
        
        mean_img_size = mean_size_folder(config_data['img_folder'])
        resize_box = [int(mean_img_size[0]*config_data['scale']),
                    int(mean_img_size[1]*config_data['scale'])]

        if config_data['scale']!=1:
            transfoms_paras.append(transforms.Resize(resize_box, interpolation=transforms.InterpolationMode.BICUBIC))

        loader = transforms.Compose(transfoms_paras)

        # create tensor dataset
        train_ims = []
        train_labels = []
        for img_id in config_data['img_ids']:
            img_path = os.path.join(config_data['img_folder'], img_id)
            train_im = loader(Image.open(img_path).convert('RGB'))
            train_label = tensor([0])

            train_ims.append(train_im.numpy())
            train_labels.append(train_label.numpy())
        
        train_ims = torch.from_numpy(np.array(train_ims))
        train_labels = torch.from_numpy(np.array(train_labels))

        print ('Training Tensor Shape is' + str(train_ims.shape))

        train_ds = DataLoader(TensorDataset(train_ims,train_labels))

        return train_ds, resize_box

    def trainModel(self):
        # load config file 
        for config_file in tqdm(os.listdir(self.config_dir)):
            if config_file.split('.')[-1] == 'json':
                with open(os.path.join(self.config_dir, config_file)) as json_file:
                    config_data = json.load(json_file)
                    train_ds, resize_box = self.genDataSet(config_data)

                    # train model 
                    model = PatchCore(f_coreset=.10, 
                                        backbone_name="wide_resnet50_2")

                    tobesaved = model.fit(train_ds)
                    
                    # save model
                    model_path = os.path.join(self.model_dir,config_file.split('.')[0])
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    train_tar = os.path.join(model_path,'Core_Path.tar')
                    train_path = os.path.join(model_path, 'Core_Path')
                    torch.save(tobesaved, train_tar)
                    torch.save(model.state_dict(), train_path)

                    # save training info
                    config_data['resize_box']=resize_box

                    json_filePath = os.path.join(model_path, 'loader_info.json')
                    json_string = json.dumps(config_data)
                    with open(json_filePath, 'w') as outfile:
                        outfile.write(json_string)

        print('finish training')
        return self.model_dir

if __name__ == "__main__":
    print ('test model training')
    config_dir = './TwoStageCore/exp/2022_07_01_10_55_40/config'
    train_session = TrainPatchCore(config_dir)
    model_dir = train_session.trainModel()
    print (model_dir)