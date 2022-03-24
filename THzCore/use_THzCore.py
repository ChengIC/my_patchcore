from models import PatchCore
import os 
import torch
import warnings # for some torch warnings regarding depreciation
import json
from PIL import Image
from torchvision import transforms
from torch import tensor
from cores_utils import AnomalyToBBox, genTimeStamp, WriteDetectImage
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
	
	def __init__(self,model_dir1,model_dir2):
		self.model_dir1 = model_dir1
		self.model_dir2 = model_dir2
		
		self.first_model = PatchCore(
				f_coreset=.10, 
				backbone_name="wide_resnet50_2",
			)

		self.second_model = PatchCore(
				f_coreset=.25, 
				backbone_name="wide_resnet50_2",
			)

		self.exp_dir = './THzCore/runs/' + genTimeStamp()
		if not os.path.exists(self.exp_dir):
			os.makedirs (self.exp_dir)

		self.first_model, self.model_paras1, self.resize_box1 = load_model_from_dir(self.first_model, self.model_dir1)
		self.second_model , self.model_paras2, self.resize_box2 = load_model_from_dir(self.second_model, self.model_dir2)
		self.loader1 = loader_from_resize(self.resize_box1)
		self.loader2 = loader_from_resize(self.resize_box2)

	def inference(self, imPath, annotation_folder=None,
				threshold_1st_stage=0.75, threshold_2nd_stage=0.75):

		image = Image.open(imPath).convert('RGB')
		copy_image = image.copy()
		original_size_width, original_size_height = image.size
		image = self.loader1(image).unsqueeze(0)
		test_img_tensor = image.to('cpu', torch.float)

		log_info = {
			'model_1st_info':self.model_dir1,
			'model_2nd_info':self.model_dir2,
			'image_path':imPath,
			'image_width':original_size_width,
			'image_height':original_size_height,
			'threshold_1st_stage':threshold_1st_stage,
			'threshold_2nd_stage':threshold_2nd_stage,
			'first_stage_img_score':0,
			'box_1st_stage':[],
			'box_2nd_stage':[],
		}

		# 1st stage
		HeatMap_Size = [original_size_height, original_size_width]
		img_lvl_anom_score, pxl_lvl_anom_score = self.first_model.inference (test_img_tensor, self.model_paras1, HeatMap_Size)
		detected_box_list_1st_stage = AnomalyToBBox(pxl_lvl_anom_score, anomo_threshold=threshold_1st_stage)
		log_info['first_stage_img_score'] = img_lvl_anom_score.item()
		
		# 2nd stage
		final_detected_list = []

		for d in detected_box_list_1st_stage:

			normalize_d = [
				d[0]/original_size_width,
				d[1]/original_size_height,
				d[2]/original_size_width,
				d[3]/original_size_height,
			]

			log_info['box_1st_stage'].append(d.tolist())

			image_patch = copy_image.crop((d[0],d[1],d[2],d[3]))
			patch_size_width, patch_size_height = image_patch.size
			Patch_HeatMap_Size = [patch_size_height, patch_size_width]
			image_patch = self.loader1(image_patch).unsqueeze(0)
			image_patch_tensor = image_patch.to('cpu', torch.float)
			_, pixel_score = self.second_model.inference (image_patch_tensor, self.model_paras1, Patch_HeatMap_Size)
			detected_box_list_2nd_stage = AnomalyToBBox(pixel_score, anomo_threshold=threshold_2nd_stage)
			if len(detected_box_list_2nd_stage)>1:
				continue
			for d2 in detected_box_list_2nd_stage:
				normalize_d2 = [
						d2[0]/patch_size_width,
						d2[1]/patch_size_height,
						d2[2]/patch_size_width,
						d2[3]/patch_size_height,
				]
				final_d2 = [
					int(normalize_d2[0]*normalize_d[0]*original_size_width),
					int(normalize_d2[1]*normalize_d[1]*original_size_height),
					int(normalize_d2[2]*normalize_d[2]*original_size_width),
					int(normalize_d2[3]*normalize_d[3]*original_size_height),
				]
				final_detected_list.append(final_d2)
				log_info['box_2nd_stage'].append(final_d2)
		
		# log 
		print (log_info)
		json_file_name = imPath.split('/')[-1].split('.')[0] + '.json'
		print (json_file_name)
		json_filePath = os.path.join(self.exp_dir, json_file_name)
		json_string = json.dumps(log_info)
		with open(json_filePath, 'w') as outfile:
			outfile.write(json_string)
		
		# write image
		if annotation_folder != None:
			image_name = imPath.split('/')[-1]
			output_img_path = os.path.join(self.exp_dir, image_name)
			WriteDetectImage(imPath,annotation_folder,final_detected_list,
								image_name,output_img_path)

if __name__ == "__main__":
	
	abnormal_folder= './datasets/full_body/test/objs'
	model_dir1 =  './THzCore/CoreModels/2022_03_23_10_05_47/first_model'
	model_dir2 = './THzCore/CoreModels/2022_03_23_10_05_47/sec_model'

	mycore = useTHzCore(model_dir1,model_dir2)
	# mycore.load_cores(model_dir1,model_dir2)

	for im in os.listdir(abnormal_folder):
		if '.jpg' in im:
			im_path = os.path.join(abnormal_folder,im)
			mycore.inference(im_path,annotation_folder='datasets/full_body/Annotations')

	
	




		

		
		
		
