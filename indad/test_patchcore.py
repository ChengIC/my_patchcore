
import torch
import os
from models import PatchCore
from PIL import Image,ImageFilter
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
        self.median_blur_size=0
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
                self.median_blur_size=data['smooth']
                try:
                    img_resize = data['imgsz']
                    transfoms_paras.append(transforms.Resize(img_resize, 
                                                            interpolation=transforms.InterpolationMode.BICUBIC))
                except:
                    img_resize = data['original_imgsz'][::-1]
                    transfoms_paras.append(transforms.Resize(img_resize, 
                                                            interpolation=transforms.InterpolationMode.BICUBIC))
        
        self.loader=transforms.Compose(transfoms_paras)

        self.model = PatchCore(backbone_name="wide_resnet50_2",)
        state_dict_path = os.path.join(model_dir,'patchcore_path')
        tar_path = os.path.join(model_dir,'patchcore_path.tar')

        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(state_dict_path))
            self.var1, self.var2, self.var3, self.var4, self.var5, self.var6 = torch.load(tar_path)
        else:
            self.model.load_state_dict(torch.load(state_dict_path,map_location ='cpu'))
            self.var1, self.var2, self.var3, self.var4, self.var5, self.var6 = torch.load(tar_path,map_location ='cpu')
            
        self.result = {}
        self.config_name = model_dir.split('/')[-1]
        self.output_img_folder, self.output_json_folder = saveResultPath(self.config_name,TimeStamp=TimeStamp)
        
    def run(self,imgs_folder,writeImage=False):
        
        for image_name in os.listdir(imgs_folder):
            if '.jpg' in image_name:
                self.result['image_name'] = image_name
                print ('Process image: ' + image_name)
                image_path = os.path.join(imgs_folder,image_name)
                image = Image.open(image_path).convert('RGB')

                if self.median_blur_size!=0:
                    image = image.filter(ImageFilter.MedianFilter(size=self.median_blur_size))
                    print ('Applying median filter on inference image with degree of '+ str(self.median_blur_size))

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