

import torch
from PIL import Image
from models import PatchCore
from ryan_utils import loader, WriteOverlayImage, setOutPutFolder, AnomalyToBBox, WriteDetectImage
import os
import csv
import warnings
import numpy as np
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
imgs_folder = './datasets/full_body/test/objs'
annotation_folder = './datasets/full_body/Annotations'

# model zoos
model_dirs = ['./model_zoo/patchcore_ratio_10']
model_dirs = [os.path.join('./model_zoo',m) for m in os.listdir('./model_zoo') ]
model_num = len(model_dirs)

# set output folder
config_setting = 'multi_core'
output_img_folder,output_csv_folder = setOutPutFolder(config_setting)

# setting of csv
csv_file = os.path.join(output_csv_folder,'inference_summary.csv')
csv_header = ['filename','threshold','xmin','ymin','xmax','ymax']
with open(csv_file,'w') as f2:
    writer = csv.writer(f2)
    writer.writerow(csv_header)


for image_name in os.listdir(imgs_folder):
    if 'jpg'in image_name: 
        accumulated_pixel_scores = None
        accumulated_img_scores = None
        for model_dir in model_dirs:

            # Load single model
            model, var1, var2, var3, var4, var5, var6 = LoadModelFromDir(model_dir)
            print ('load model' + model_dir)

            # Load Inference Image
            image_path = os.path.join(imgs_folder,image_name)
            image = Image.open(image_path).convert('RGB')
            image = loader(image).unsqueeze(0)
            test_img_tensor = image.to('cpu', torch.float)
            
            # PatchCore Inference
            HeatMap_Size = [test_img_tensor.shape[2], test_img_tensor.shape[3]]
            img_lvl_anom_score, pxl_lvl_anom_score = model.inference(test_img_tensor, var1, var2, var3, var4, HeatMap_Size, var6 )
            
            if accumulated_pixel_scores is None:
                accumulated_pixel_scores = pxl_lvl_anom_score.numpy()
            else:
                accumulated_pixel_scores = np.add(accumulated_pixel_scores,pxl_lvl_anom_score.numpy())
            
            if accumulated_img_scores is None:
                accumulated_img_scores = img_lvl_anom_score.numpy()
            else:
                accumulated_img_scores = np.add(accumulated_img_scores,img_lvl_anom_score.numpy())

        # accumulated result processing
        accumulated_pixel_scores = torch.from_numpy(accumulated_pixel_scores/model_num)
        accumulated_img_scores = torch.as_tensor(accumulated_img_scores/model_num)

        # Anomaly Region to bbox
        detected_box_list = AnomalyToBBox(accumulated_pixel_scores, anomo_threshold=0.75, x_ratio=1, y_ratio=1)

        if write_image:
            WriteOverlayImage(image_path,output_img_folder,image_name,
                            test_img_tensor,accumulated_img_scores,pxl_lvl_anom_score)
            WriteDetectImage(image_path, annotation_folder,detected_box_list,image_name,output_img_folder)

        if write_result_csv:
            with open(csv_file,'a') as f2:
                writer = csv.writer(f2)
                for bbox in detected_box_list:
                    csv_data = [image_name, 0.75, bbox[0], bbox[1], bbox[2], bbox[3]]
                    writer.writerow(csv_data)
