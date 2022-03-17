from models import PatchCore
from train_utils import genDS
import torch
import os
import warnings # for some torch warnings regarding depreciation
warnings.filterwarnings("ignore")

def train_model(train_folder):
    model = PatchCore(
                    f_coreset=.10, 
                    backbone_name="wide_resnet50_2",
                )
    dataGS = genDS(train_folder)
    train_ds = dataGS.genTrainDS(0.1)

    tobesaved = model.fit(train_ds)

    save_model_dir = './model_zoos_regional/'
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    train_tar = os.path.join(save_model_dir + '_' + 'THzCore_Path.tar')
    train_path = os.path.join(save_model_dir + '_' + 'THzCore_Path')

    torch.save(tobesaved, train_tar)
    torch.save(model.state_dict(), train_path)
    

if __name__ == "__main__":
    train_folder =  './datasets/region_body/leg'
    train_model(train_folder)
