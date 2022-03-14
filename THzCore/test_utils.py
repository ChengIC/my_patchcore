import os
from models import PatchCore
from PIL import Image
import torch
from torch.utils.data import DataLoader,TensorDataset
from torch import tensor
from torchvision import transforms

class laod_THzCore():

    def __init__(self,save_model_dir = './model_zoos_regional'):
        self.region_types = ['leg','arm','main_body']
        
        self.model_dict = {}
        for region_type in self.region_types:

            tar_path = os.path.join(save_model_dir,region_type + '_' + 'THzCore_Path.tar')
            state_dict_path = os.path.join(save_model_dir, region_type + '_' + 'THzCore_Path')

            model = PatchCore(backbone_name="wide_resnet50_2",)

            if torch.cuda.is_available():
                model.load_state_dict(torch.load(state_dict_path))
                var1, var2, var3, var4, var5, var6 = torch.load(tar_path)
            else:
                model.load_state_dict(torch.load(state_dict_path,map_location ='cpu'))
                var1, var2, var3, var4, var5, var6 = torch.load(tar_path,map_location ='cpu')

            model_paras = [var1, var2, var3, var4, var5, var6]
            self.model_dict[region_type]={
                'model':model,
                'model_paras':model_paras
            }

    
    def inference_img(self,img_path,img_type):
        if img_type == 'main_body':
            resize_box = [136,280]
        elif img_type == 'arm':
            resize_box = [100,182]
        elif img_type == 'leg':
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
        loader=transforms.Compose(transfoms_paras)

        if img_type not in self.region_types:
            raise Exception('wrong image type')

        model = self.model_dict[img_type]['model']
        var1, var2, var3, var4, _, var6 = self.model_dict[img_type]['model_paras']

        image = Image.open(img_path).convert('RGB')
        original_size_width, original_size_height = image.size
        image = loader(image).unsqueeze(0)
        test_img_tensor = image.to('cpu', torch.float)
        HeatMap_Size = [original_size_height, original_size_width]
        img_lvl_anom_score, pxl_lvl_anom_score = model.inference(test_img_tensor, var1, var2, var3, var4, HeatMap_Size, var6)

        return img_lvl_anom_score, pxl_lvl_anom_score 


if __name__ == "__main__":
        save_model_dir = './model_zoos_regional'
        THzCore = laod_THzCore(save_model_dir)
        img_folder = './datasets/region_body/my_test'
        for img_file in os.listdir(img_folder):
            if 'jpg' in img_file:
                img_path = os.path.join(img_folder,img_file)
                img_type = img_file.split('_')[0]
                img_type = img_type.replace(' ','_')
                img_lvl_anom_score, pxl_lvl_anom_score = THzCore.inference_img(img_path,img_type)
                print (img_lvl_anom_score, pxl_lvl_anom_score)

