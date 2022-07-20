
from ran_utils import *
import os 
import json
from models import PatchCore
from tqdm import tqdm
import torch
from torch import tensor
import os
from PIL import Image
from torchvision import transforms



def load_model_from_dir(model_path):
    state_dict_path = os.path.join(model_path,'Core_Path')
    tar_path = os.path.join(model_path,'Core_Path.tar')
    configPath =os.path.join(model_path,'loader_info.json')

    # load model and config
    model = PatchCore(f_coreset=.10, backbone_name="wide_resnet50_2")
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(state_dict_path))
        model_paras = torch.load(tar_path)
    else:
        model.load_state_dict(torch.load(state_dict_path,map_location ='cpu'))
        model_paras = torch.load(tar_path,map_location ='cpu')
    with open(configPath) as json_file:
        config_data = json.load(json_file)

    print ('loading model dir: ' + model_path + ' sucessfully')

    return model, model_paras, config_data

def load_image_to_tensor_cpu(img_path, config_data):

    IMAGENET_MEAN = tensor([.485, .456, .406])
    IMAGENET_STD = tensor([.229, .224, .225])
    transfoms_paras = [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    if config_data['scale']!=1:
        transfoms_paras.append(transforms.Resize(config_data['resize_box'], interpolation=transforms.InterpolationMode.BICUBIC))
    loader = transforms.Compose(transfoms_paras)

    image = Image.open(img_path).convert('RGB')
    original_size_width, original_size_height = image.size
    image = loader(image).unsqueeze(0)

    test_img_tensor = image.to('cpu', torch.float)
    HeatMap_Size = [original_size_height, original_size_width]

    return test_img_tensor, HeatMap_Size


class InferenceCore():

    def __init__(self, model_path=None, run_dir=None) :
        self.model_path = model_path
        self.model_name = self.model_path.split('/')[-1]
        self.run_dir = run_dir
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        # begin loading patch-core from model path
        self.model, self.model_paras, self.config_data = load_model_from_dir(model_path)

    def inference_one_img(self, img_path):
        # for testing and insight
        
        test_img_tensor, HeatMap_Size = load_image_to_tensor_cpu(img_path, self.config_data)
        img_lvl_anom_score, pxl_lvl_anom_score = self.model.inference (test_img_tensor, self.model_paras, HeatMap_Size)

        result  = {
            'img_path': img_path,
            'img_score':img_lvl_anom_score.tolist(),
            'pixel_score':pxl_lvl_anom_score.tolist(),
        }

        return result, self.run_dir
