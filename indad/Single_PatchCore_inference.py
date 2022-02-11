

import torch
from PIL import Image
from models import PatchCore
from ryan_utils import loader, WriteOverlayImage, setOutPutFolder, AnomalyToBBox,readXML
import os
import csv
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
compare_annotation = True
write_result_csv = True

model_dir = './model_zoo/patchcore_ratio_10'
model, var1, var2, var3, var4, var5, var6 = LoadModelFromDir(model_dir)
config_setting = 'patchcore_ratio_10'
my_folder = './datasets/full_body/test/objs'
output_img_folder,output_csv_folder = setOutPutFolder(config_setting)

# setting of csv
csv_file = os.path.join(output_csv_folder,'inference_summary.csv')
csv_header = ['filename','threshold','xmin','ymin','xmax','ymax']
with open(csv_file,'w') as f2:
    writer = csv.writer(f2)
    writer.writerow(csv_header)

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
        
        # Anomaly Region to bbox
        detected_box_list = AnomalyToBBox(pxl_lvl_anom_score, anomo_threshold=0.75, x_ratio=1, y_ratio=1)

        if write_image:
            WriteOverlayImage(image_path,output_img_folder,image_name,
                            test_img_tensor,img_lvl_anom_score,pxl_lvl_anom_score)
        
        if write_result_csv:
            with open(csv_file,'a') as f2:
                writer = csv.writer(f2)
                for bbox in detected_box_list:
                    csv_data = [image_name, 0.75, bbox[0], bbox[1], bbox[2], bbox[3]]
                    writer.writerow(csv_data)