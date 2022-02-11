

import torch
from PIL import Image
from models import PatchCore
from ryan_utils import loader, WriteOverlayImage, setOutPutFolder, AnomalyToBBox
import os 
import warnings
warnings.filterwarnings("ignore")


def LoadModelFromDir(model_dir,method='patchcore'):
    if method == 'patchcore':
        model = PatchCore(
                f_coreset=0.1, 
                backbone_name="wide_resnet50_2",
             )
        state_dict_path = os.path.join(model_dir,'patchcore_path')
        tar_path = os.path.join(model_dir,'patchcore_path.tar')
        model.load_state_dict(torch.load(state_dict_path))
        var1, var2, var3, var4, var5, var6 = torch.load(tar_path)
        return model, var1, var2, var3, var4, var5, var6 
    else:
        pass

write_image = True
model_dir = './model_zoo/patchcore_ratio_10'
model, var1, var2, var3, var4, var5, var6 = LoadModelFromDir(model_dir)
config_setting = 'patchcore_ratio_10'
my_folder = './datasets/full_body/test/objs'
output_folder = setOutPutFolder(config_setting)

for image_name in os.listdir(my_folder):
    if 'jpg'in image_name:
        # Load Inference Image
        image_path = os.path.join(my_folder,image_name)
        image = Image.open(image_path).convert('RGB')
        image = loader(image).unsqueeze(0)
        test_img_tensor = image.to('cpu', torch.float)
        
        # PatchCore Inference
        HeatMap_Size = [test_img_tensor.shape[2], test_img_tensor.shape[3]]
        img_lvl_anom_score, pxl_lvl_anom_score = model.inference(test_img_tensor, var1, var2, var3, var4, HeatMap_Size, var6 )
        
        if write_image == True:
            WriteOverlayImage(image_path,output_folder,image_name,
                            test_img_tensor,img_lvl_anom_score,pxl_lvl_anom_score)

        detected_box_list = AnomalyToBBox(pxl_lvl_anom_score, anomo_threshold=0.75, x_ratio=1, y_ratio=1)
        print (detected_box_list)