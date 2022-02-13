'''

Training PatchCore model with selected group images

'''


import click
from PIL import Image
from models import SPADE, PaDiM, PatchCore
from torch.utils.data import DataLoader,TensorDataset
import torch
from torchvision import transforms
from torch import tensor
import os
import warnings # for some torch warnings regarding depreciation
import time
import random
import time
import numpy 

warnings.filterwarnings("ignore")

all_normal_image_path = './datasets/full_body/train/good/'

ALLOWED_METHODS = ["spade", "padim", "patchcore"]

IMAGENET_MEAN = tensor([.485, .456, .406])
IMAGENET_STD = tensor([.229, .224, .225])
loader=transforms.Compose([
                # transforms.Resize([224,224], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

def init_model(method):
    if method == "spade":
            model = SPADE(
                k=50,
                backbone_name="wide_resnet50_2",
            )
    elif method == "padim":
            model = PaDiM(
                d_reduced=350,
                backbone_name="wide_resnet50_2",
            )
    elif method == "patchcore":
            model = PatchCore(
                f_coreset=.20, 
                backbone_name="wide_resnet50_2",
            )
    return model

def genTrainDS(img_folder,train_text='normal_train_samples.txt'):
    train_ims = []
    train_labels = []

    with open(train_text,'r') as f:
        train_img_name = f.readlines()
        for img_name in train_img_name:
            img_path = os.path.join(img_folder, img_name.strip('\n'))
            
            train_im = Image.open(img_path).convert('RGB')
            train_im = loader(train_im)
            train_label = tensor([0])
            train_ims.append(train_im.numpy())
            train_labels.append(train_label.numpy())

    train_ims = numpy.array(train_ims)
    train_labels = numpy.array(train_labels)
    
    print (train_ims.shape)

    train_ims = torch.from_numpy(train_ims)
    train_labels = torch.from_numpy(train_labels)

    train_data = TensorDataset(train_ims,train_labels)
    train_ds = DataLoader(train_data)
    
    return train_ds

def genConfigs(img_folder,fixed_seed=2022):
    config_folder = './config_shuffle'
    if not os.path.exists(config_folder):
        os.makedirs (config_folder)
    img_filenames = os.listdir(img_folder)
    random.seed(fixed_seed)
    set_nums = 21
    for r in range(set_nums):
        random.shuffle(img_filenames)
        train_filenames = img_filenames[0:10]
        config_text_file = os.path.join(config_folder, 'set_' + str(r) + '.txt' )
        with open(config_text_file,'w') as f:
            for ele in train_filenames:
                f.write(ele + "\n")
    config_list = os.listdir(config_folder)
    return config_folder, config_list


def savePath(method_name,config_filename):
    model_path = './model_zoo/' + method_name + '_' + config_filename.strip('.txt') 
    if not os.path.exists(model_path):
        os.makedirs (model_path)
    train_tar = os.path.join(model_path, 'patchcore_path.tar')
    train_path = os.path.join(model_path, 'patchcore_path')
    return train_tar, train_path


def run_model(method: str):

    print(f"\n█│ Running {method}.")
    
    img_folder = './datasets/full_body/train/good/'
    
    config_folder, config_list = genConfigs(img_folder)
    
    for config_filename in config_list:
        print("   Training with setting of " + config_filename)
        config_filepath = os.path.join(config_folder,config_filename)
        # print (config_filepath)
        train_ds = genTrainDS(img_folder,config_filepath)

        ### training
        model = init_model(method)
        train_tar, train_path = savePath(method,config_filename)
        tobesaved = model.fit(train_ds)

        # save the model path
        torch.save(tobesaved, train_tar)
        torch.save(model.state_dict(), train_path)
        
    print ('Finish Group Image Training')

@click.command()
@click.option("--method",default = "patchcore")
# @click.option("--dataset", default="full_body", help="Dataset, defaults to all datasets.")
def cli_interface(method: str): 
   
    method = method.lower()

    run_model(method)
    

if __name__ == "__main__":
    cli_interface()
