from PIL import Image
from models import PatchCore
from train_utils import genDS
import torch
import numpy as np
import os
import warnings # for some torch warnings regarding depreciation
warnings.filterwarnings("ignore")

def train_model(train_folder,region_type):
    model = PatchCore(
                    f_coreset=.10, 
                    backbone_name="wide_resnet50_2",
                )
    dataGS = genDS(region_type)
    train_ds = dataGS.genTrainDS(train_folder,0.01)

    tobesaved = model.fit(train_ds)

    save_model_dir = './model_zoos_regional/'
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    train_tar = os.path.join(save_model_dir,region_type + '_' + 'THzCore_Path.tar')
    train_path = os.path.join(save_model_dir, region_type + '_' + 'THzCore_Path')

    torch.save(tobesaved, train_tar)
    torch.save(model.state_dict(), train_path)
    

if __name__ == "__main__":
    train_folders = [   './datasets/region_body/leg',
                        './datasets/region_body/arm',
                        './datasets/region_body/main_body'
    ]
    for train_folder in train_folders:
        region_type = train_folder.split('/')[-1]
        train_model(train_folder,region_type)
