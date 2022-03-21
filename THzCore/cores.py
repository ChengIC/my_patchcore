from models import PatchCore
from train_utils import genDS
import os 
import torch
from cores_utils import genTimeStamp, SplitList, Image2AnomoBox, readXML,IoU,savePatchImg
import warnings # for some torch warnings regarding depreciation
warnings.filterwarnings("ignore")
import json

class THzCore():
	
	def __init__(self,normal_imgs_dir,abnormal_imgs_dir):
		self.first_model = PatchCore(
				f_coreset=.10, 
				backbone_name="wide_resnet50_2",
			)

		self.second_model = PatchCore(
				f_coreset=.25, 
				backbone_name="wide_resnet50_2",
			)
		self.normal_imgs_dir=normal_imgs_dir
		self.abnormal_imgs_dir=abnormal_imgs_dir

	def train_1st_stage(self):
		firstDS = genDS(training_folder=self.normal_imgs_dir,
						resize_box=None)

		percentage =  2/len(os.listdir(self.normal_imgs_dir))
		if percentage>1:
			train_ds,self.ds_loader, resize_box= firstDS.genTrainDS()
		else:
			train_ds,self.ds_loader, resize_box = firstDS.genTrainDS(percentage)
		self.tobesaved = self.first_model.fit(train_ds)

		self.models_dir = './THzCore/CoreModels/' + genTimeStamp() 
		if not os.path.exists(self.models_dir):
			os.makedirs (self.models_dir)

		self.models_dir1 = os.path.join(self.models_dir,'first_model')
		if not os.path.exists(self.models_dir1):
			os.makedirs (self.models_dir1)
		train_tar = os.path.join(self.models_dir1, 'Core_Path.tar')
		train_path = os.path.join(self.models_dir1, 'Core_Path')
		torch.save(self.tobesaved, train_tar)
		torch.save(self.first_model.state_dict(), train_path)

		json_filePath = os.path.join(self.models_dir1, 'loader_info.json')
		data_dict = {'box':resize_box}
		json_string = json.dumps(data_dict)
		with open(json_filePath, 'w') as outfile:
			outfile.write(json_string)

	def prep_2nd_stage(self,annotation_dir):
		print ('Prepare data for 2nd stage training')
		normal_imgs = SplitList(self.normal_imgs_dir,percentage=0.02)
		abnormal_imgs = SplitList(self.abnormal_imgs_dir,percentage=0.02)
		
		self.second_img_paths = os.path.join(self.models_dir,'sec_imgs')
		if not os.path.exists(self.second_img_paths):
			os.makedirs (self.second_img_paths)

		self.objs_img_paths = os.path.join(self.models_dir,'obj_imgs')
		if not os.path.exists(self.objs_img_paths):
			os.makedirs (self.objs_img_paths)

		for n_im in normal_imgs:
			n_im_a_box_list = Image2AnomoBox(n_im,self.ds_loader,self.first_model,self.tobesaved)
			if len(n_im_a_box_list)>0:
				for bbox in n_im_a_box_list:
					savePatchImg(n_im,bbox,self.second_img_paths)

		for a_im in abnormal_imgs:
			a_im_a_box_list = Image2AnomoBox(a_im, self.ds_loader,self.first_model,self.tobesaved)
			image_name = a_im.split('/')[-1]
			box_dict = readXML(annotation_dir, image_name)
			for a_r in a_im_a_box_list:
				for obj in box_dict:
					bbox0, bbox1 = a_r, box_dict[obj]
					if IoU (bbox0, bbox1)==0:
						savePatchImg(a_im,bbox0,self.second_img_paths)
					if IoU(bbox0, bbox1)>0:
						savePatchImg(a_im,bbox0,self.objs_img_paths)
		
	def train_2nd_stage(self):
		print ('Begin 2nd stage training')
		secDS = genDS(training_folder=self.second_img_paths,
				resize_box=None)

		percentage =  20/len(os.listdir(self.normal_imgs_dir))
		if percentage>1:
			train_ds2, self.ds_loader, resize_box = secDS.genTrainDS()
		else:
			train_ds2, self.ds_loader, resize_box = secDS.genTrainDS(percentage)

		self.tobesaved2 = self.second_model.fit(train_ds2)

		self.models_dir2 = os.path.join(self.models_dir,'sec_model')
		if not os.path.exists(self.models_dir2):
			os.makedirs (self.models_dir2)
		train_tar = os.path.join(self.models_dir2, 'Core_Path.tar')
		train_path = os.path.join(self.models_dir2, 'Core_Path')
		torch.save(self.tobesaved2, train_tar)
		torch.save(self.second_model.state_dict(), train_path)
		json_filePath = os.path.join(self.models_dir2, 'loader_info.json')
		data_dict = {'box':resize_box}
		json_string = json.dumps(data_dict)
		with open(json_filePath, 'w') as outfile:
			outfile.write(json_string)

if __name__ == "__main__":
	normal_folder =  './datasets/THz_Body/train/good'
	abnormal_folder= './datasets/full_body/test/objs'
	annotation_dir ='./datasets/full_body/Annotations'
	mycore = THzCore(normal_folder,abnormal_folder)
	mycore.train_1st_stage()
	mycore.prep_2nd_stage(annotation_dir)
	mycore.train_2nd_stage()
	




		

		
		
		
