
import torch
import os
from models import PatchCore
from PIL import Image
from torch import tensor
from torchvision import transforms
from save_utils import saveResultPath
import json
import warnings # for some torch warnings regarding depreciation
from draw_utils import WriteOverlayImage
warnings.filterwarnings("ignore")

class RunPatchcore():
    def __init__(self,model_dir,resize=None,center_crop=None,
                                configPath=None,TimeStamp=None):
        self.reszie = None
        IMAGENET_MEAN = tensor([.485, .456, .406])
        IMAGENET_STD = tensor([.229, .224, .225])
        transfoms_paras = [
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                    ]
        if resize!=None:
            transfoms_paras.append(transforms.Resize(resize, 
                                                    interpolation=transforms.InterpolationMode.BICUBIC))
        if center_crop!=None:
            transfoms_paras.append(transforms.CenterCrop(center_crop))
        
        if configPath!=None:
            with open(configPath) as json_file:
                data = json.load(json_file)
                img_resize = data['imgsz'][::-1]
                transfoms_paras.append(transforms.Resize(img_resize, 
                                                    interpolation=transforms.InterpolationMode.BICUBIC))
        
        self.loader=transforms.Compose(transfoms_paras)

        self.model = PatchCore(backbone_name="wide_resnet50_2",)
        state_dict_path = os.path.join(model_dir,'patchcore_path')
        tar_path = os.path.join(model_dir,'patchcore_path.tar')
        self.model.load_state_dict(torch.load(state_dict_path))
        self.var1, self.var2, self.var3, self.var4, self.var5, self.var6 = torch.load(tar_path)
        
        self.result = {}
        if configPath is None:
            self.configPath = model_dir.split('/')[-1]
        else:
            self.configPath = configPath

        self.output_img_folder, self.output_json_folder = saveResultPath(self.configPath,TimeStamp=TimeStamp)
        
    def run(self,imgs_folder,writeImage=False):
        
        for image_name in os.listdir(imgs_folder):
            self.result['image_name'] = image_name
            image_path = os.path.join(imgs_folder,image_name)
            image = Image.open(image_path).convert('RGB')
            original_size_width, original_size_height = image.size
            image = self.loader(image).unsqueeze(0)
            test_img_tensor = image.to('cpu', torch.float)

            # PatchCore Inference
            HeatMap_Size = [original_size_height, original_size_width]
            
            img_lvl_anom_score, pxl_lvl_anom_score = self.model.inference(test_img_tensor, self.var1, self.var2,
                                                                            self.var3, self.var4, HeatMap_Size, self.var6)
            self.result['image_score'] = img_lvl_anom_score.numpy().tolist()
            self.result['pxl_lvl_anom_score'] = pxl_lvl_anom_score.numpy().tolist()
            json_string = json.dumps(self.result)
            json_filePath = os.path.join(self.output_json_folder,image_name.split('.')[0]+'.json')
            with open(json_filePath, 'w') as outfile:
                outfile.write(json_string)
            
            if writeImage:
                output_img_path = os.path.join(self.output_img_folder,image_name)
                WriteOverlayImage(image_path,None,img_lvl_anom_score,
                                    pxl_lvl_anom_score,output_img_path)
            
        print('finish inferencing')
        return self.output_img_folder, self.output_json_folder

if __name__ == "__main__":
    configPath = './config/semi/percentage_0.2_ZRBX5NKS.json'
    model_dir = 'model_zoo/percentage_0.2_ZRBX5NKS/2022_02_16_15_46_29/'
    run1 = RunPatchcore(model_dir,configPath=configPath)
    test_imgs_folder = './datasets/full_body/test/objs'
    run1.run(test_imgs_folder,writeImage=True)