
from models import PatchCore
from use_THzCore import load_model_from_dir,loader_from_resize
from PIL import Image
import torch
from cores_utils import AnomalyToBBox, IoU, readXML,genTimeStamp,WriteDetectImage
import os 

IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']  # include image suffixes

# data
abnormal_folder = './datasets/full_body/test/objs'
annotation_folder = './datasets/full_body/Annotations'

# init folder
time_string = genTimeStamp()
failed_detect_dir = './THzCore/runs/' + time_string + '/failed_detect_dir'
parts_detect_dir = './THzCore/runs/' + time_string + '/parts_detect_dir'
full_detect_dir = './THzCore/runs/' + time_string + '/full_detect_dir'
if not os.path.exists(failed_detect_dir):
	os.makedirs (failed_detect_dir)
if not os.path.exists(parts_detect_dir):
	os.makedirs (parts_detect_dir)
if not os.path.exists(full_detect_dir):
	os.makedirs (full_detect_dir)

# load first stage model
model_dir1 = './THzCore/CoreModels/exp_1st_model'
first_model = PatchCore(
			f_coreset=.10, 
			backbone_name="wide_resnet50_2",
			)

first_model, model_paras1, resize_box1 = load_model_from_dir(first_model,model_dir1)
loader1 = loader_from_resize(resize_box1)

for image_name in os.listdir(abnormal_folder):
	if image_name.split('.')[-1] in IMG_FORMATS:
		imPath = os.path.join(abnormal_folder,image_name)
		image = Image.open(imPath).convert('RGB')
		original_size_width, original_size_height = image.size
		image = loader1(image).unsqueeze(0)
		test_img_tensor = image.to('cpu', torch.float)

		HeatMap_Size = [original_size_height, original_size_width]
		img_lvl_anom_score, pxl_lvl_anom_score = first_model.inference (test_img_tensor, model_paras1, HeatMap_Size)
		detected_box_list_1st_stage = AnomalyToBBox(pxl_lvl_anom_score, anomo_threshold=0.75)

		box_dict = readXML(annotation_folder, image_name)
		score = {}
		for bbox in detected_box_list_1st_stage:
			for cls in box_dict:
				if IoU(bbox,box_dict[cls])>0:
					score[cls]=True

		if len(score)==0:
			output_dir=failed_detect_dir
		elif len(score)>0 and len(score)<len(box_dict):
			output_dir=parts_detect_dir
		elif len(score)>0 and len(score)==len(box_dict):
			output_dir=full_detect_dir
		else:
			print ('BUG BUG BUG BUG')
		
		output_img_path = os.path.join(output_dir,image_name)
		WriteDetectImage(imPath,annotation_folder,detected_box_list_1st_stage,
                            image_name,output_img_path)



