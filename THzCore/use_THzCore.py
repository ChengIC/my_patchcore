from models import PatchCore
import os 
import torch
import warnings # for some torch warnings regarding depreciation
import json
from PIL import Image
from torchvision import transforms
from torch import tensor
from cores_utils import AnomalyToBBox
warnings.filterwarnings("ignore")


def load_model_from_dir(model, dir):
	state_dict_path = os.path.join(dir,'Core_Path')
	tar_path = os.path.join(dir,'Core_Path.tar')
	configPath = os.path.join(dir,'loader_info.json')

	if torch.cuda.is_available():
		model.load_state_dict(torch.load(state_dict_path))
		model_paras = torch.load(tar_path)
	else:
		model.load_state_dict(torch.load(state_dict_path,map_location ='cpu'))
		model_paras = torch.load(tar_path,map_location ='cpu')
	
	with open(configPath) as json_file:
			data = json.load(json_file)
			resize_box=data['box']

	return model, model_paras, resize_box

def loader_from_resize(resize_box):
	IMAGENET_MEAN = tensor([.485, .456, .406])
	IMAGENET_STD = tensor([.229, .224, .225])
	transfoms_paras = [
			transforms.Resize(resize_box, interpolation=transforms.InterpolationMode.BICUBIC),
			transforms.ToTensor(),
			transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
	]
	return transforms.Compose(transfoms_paras)

class useTHzCore():
	
	def __init__(self):
		self.first_model = PatchCore(
				f_coreset=.10, 
				backbone_name="wide_resnet50_2",
			)

		self.second_model = PatchCore(
				f_coreset=.25, 
				backbone_name="wide_resnet50_2",
			)
	
	def load_cores(self,model_dir1,model_dir2):
		self.first_model, self.model_paras1, self.resize_box1 = load_model_from_dir(self.first_model, model_dir1)
		self.second_model , self.model_paras2, self.resize_box2 = load_model_from_dir(self.second_model, model_dir2)
		self.loader1 = loader_from_resize(self.resize_box1)
		self.loader2 = loader_from_resize(self.resize_box2)

	def inference(self,imPath):
		image = Image.open(imPath).convert('RGB')
		copy_image = image.copy()
		original_size_width, original_size_height = image.size
		image = self.loader1(image).unsqueeze(0)
		test_img_tensor = image.to('cpu', torch.float)

		HeatMap_Size = [original_size_height, original_size_width]
		_, pxl_lvl_anom_score = self.first_model.inference (test_img_tensor, self.model_paras1, HeatMap_Size)
		detected_box_list = AnomalyToBBox(pxl_lvl_anom_score, anomo_threshold=0.75)
		for d in detected_box_list:
			image_patch = copy_image.crop((d[0],d[1],d[2],d[3]))
			original_size_width, original_size_height = image_patch.size
			image_patch = self.loader1(image_patch).unsqueeze(0)
			image_patch_tensor = image_patch.to('cpu', torch.float)
			image_score, pixel_score = self.second_model.inference (image_patch_tensor, self.model_paras1, HeatMap_Size)

			print (image_score, pixel_score)



if __name__ == "__main__":
	
	abnormal_folder= './datasets/full_body/test/objs'
	
	mycore = useTHzCore()
	model_dir1 =  './THzCore/CoreModels/2022_03_17_13_12_16/first_model'
	model_dir2 = './THzCore/CoreModels/2022_03_17_13_12_16/sec_model'
	mycore.load_cores(model_dir1,model_dir2)

	for im in os.listdir(abnormal_folder):
		if '.jpg' in im:
			im_path = os.path.join(abnormal_folder,im)
			mycore.inference(im_path)

	
	




		

		
		
		
