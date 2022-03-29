from models import PatchCore
from train_utils import genDS
import os 
import torch
from cores_utils import *
import warnings # for some torch warnings regarding depreciation
warnings.filterwarnings("ignore")
import json
from config_utils import *

class MultiCores():
	def __init__(self,
				training_img_folder=None,
				mode=None,
				multicore_dir=None,
				timestring=None):
		
		self.training_img_folder=training_img_folder
		self.mode=mode
		self.multicore_dir=multicore_dir
		self.timestring = timestring
		if self.timestring==None:
			self.timestring = genTimeStamp()
		
		if self.mode=='train':
			if self.training_img_folder==None:
				raise 'Please define training folder of normal images'

		elif self.mode=='inference' or self.mode=='test':
			if self.multicore_dir == None:
				raise 'Please define model dir for inference or testing'

			# load multiple patchcore models at the same time
			self.models_dict = {}
			idx=0
			for roots, _, files in os.walk(self.multicore_dir):
				if 'Core_Path.tar' in files:
					idx+=1
					model = PatchCore(f_coreset=.10, 
									backbone_name="wide_resnet50_2")
					model, model_paras, resize_box = load_model_from_dir(model, roots)
					
					self.models_dict["model{}".format(idx)] = {
								'roots':roots,
								'model':model,
								'model_paras':model_paras,
								'resize_box':resize_box,
					}
		
		else:
			pass
	
	def train_single_core(self,config_path=None):
		# load config
		with open(config_path) as json_file:
			config_data = json.load(json_file)

		# generate training dataset
		DS = genDS(training_folder=self.training_img_folder,
					scale=config_data['scale'],
					resize_box=None)
		train_ds, _, ds_info  = DS.genTrainDS_by_config(config_data)

		# train model 
		self.model = PatchCore(f_coreset=.10, 
								backbone_name="wide_resnet50_2")
		self.tobesaved = self.model.fit(train_ds)
		
		# save model
		self.model_dir = os.path.join('./THzCore','MultiCoreModels',config_data['config_id'],self.timestring)
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		train_tar = os.path.join(self.model_dir,'Core_Path.tar')
		train_path = os.path.join(self.model_dir,'Core_Path')
		torch.save(self.tobesaved, train_tar)
		torch.save(self.model.state_dict(), train_path)

		# save data loader for inference
		json_filePath = os.path.join(self.model_dir, 'loader_info.json')
		json_string = json.dumps(ds_info)
		with open(json_filePath, 'w') as outfile:
			outfile.write(json_string)

	def train_multicores(self,config_dir):
		for config_file in os.listdir(config_dir):
			config_path = os.path.join(config_dir,config_file)
			print ('Start training with ' + config_path)
			self.train_single_core(config_path)
		

	def inference(self,img_path):
		# create exp dir
		self.exp_dir = os.path.join('./THzCore','runs','multicores',self.timestring)
		if not os.path.exists(self.exp_dir):
			os.makedirs(self.exp_dir)
		
		# inference image with pathcore one by one
		exp_info = {}
		for single_model in self.models_dict:
			roots = self.models_dict[single_model]['roots']
			model = self.models_dict[single_model]['model']
			model_paras = self.models_dict[single_model]['model_paras']
			resize_box = self.models_dict[single_model]['resize_box']
			
			# load image tensor
			loader = loader_from_resize(resize_box)
			image = Image.open(img_path).convert('RGB')
			original_size_width, original_size_height = image.size
			image = loader(image).unsqueeze(0)
			test_img_tensor = image.to('cpu', torch.float)

			# inference
			HeatMap_Size = [original_size_height, original_size_width]
			_, pxl_lvl_anom_score = model.inference (test_img_tensor, model_paras, HeatMap_Size)
			exp_info ={
				'img_path':img_path,
				'pxl_lvl_anom_score':pxl_lvl_anom_score.numpy().tolist(),
				'resize_box':resize_box,
				'roots_dir':roots,
			}
			img_id = img_path.split('/')[-1].split('.')[0]
			s = roots.split('/')[-2]
			json_file_name = img_id + '_conf_' + s + '.json' 
			json_filePath = os.path.join(self.exp_dir, json_file_name)
			json_string = json.dumps(exp_info)
			with open(json_filePath, 'w') as outfile:
				outfile.write(json_string)

	def inference_dir(self,img_dir=None):
		if img_dir == None:
			raise 'No valid img dir'

		for img_name in os.listdir(img_dir):
			img_path = os.path.join(img_dir,img_name)
			self.inference(img_path)

	def visualize(self,json_path=None):
		if json_path == None:
			raise 'No result for visualization'
		
		# create vis dir
		self.vis_dir = os.path.join('./THzCore','vis','multicores',self.timestring)
		if not os.path.exists(self.vis_dir):
			os.makedirs(self.vis_dir)
		
		# visualize overlay image from results
		output_img_path = os.path.join(self.vis_dir,json_path.split('/')[-1].split('.json')[0]+'.jpg')
		print (output_img_path)
		with open(json_path) as json_file:
			json_data = json.load(json_file)
			img_path = json_data['img_path']
			pxl_lvl_anom_score = torch.from_numpy(np.array(json_data['pxl_lvl_anom_score']))
			out_img = visOverlay(img_path,pxl_lvl_anom_score)
			out_img.save(output_img_path)

	def vis_form_dir (self,json_dir=None):
		if json_dir == None:
			raise 'No valid json dir'

		for json_name in os.listdir(json_dir):
			json_path = os.path.join(json_dir,json_name)
			try:
				self.visualize(json_path)
			except:
				pass
	
	

if __name__ == "__main__":
	normal_folder =  './datasets/full_body/train/good'
	# gen config
	config_dir = genConfig().genMultiConfig(config_idea='human_depended',normal_img_folder='./datasets/full_body/train/good')

	time_string = genTimeStamp()
	# training
	mycore = MultiCores(mode='train',training_img_folder='./datasets/full_body/train/good',timestring=time_string)
	mycore.train_multicores(config_dir)

	# inference
	mycore = MultiCores(mode='inference',multicore_dir='./THzCore/MultiCoreModels/'+time_string,timestring=time_string)
	mycore.inference_dir(img_dir='./datasets/full_body/test/objs')

	# visualize
	mycore = MultiCores()
	mycore.vis_form_dir('./THzCore/runs/multicores/'+time_string)