from models import PatchCore
from train_utils import genDS
import os 
import torch
from cores_utils import *
import warnings # for some torch warnings regarding depreciation
warnings.filterwarnings("ignore")
import json
from config_utils import *
import datetime

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
			print ('loading ' + self.multicore_dir)
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
		self.model_dir = os.path.join('./THzCore/Exp',self.timestring, 'models', config_data['config_id'])
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
		self.run_dir = os.path.join('./THzCore/Exp',self.timestring,'runs')
		if not os.path.exists(self.run_dir):
			os.makedirs(self.run_dir)
		
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
			detected_box_list = PixelScore2Boxes(pxl_lvl_anom_score)
			exp_info ={
				'img_path':img_path,
				'detected_box_list':detected_box_list,
				'resize_box':resize_box,
				'roots_dir':roots,
			}
			img_id = img_path.split('/')[-1].split('.')[0]
			s = roots.split('/')[-1]
			json_file_name = img_id + '_config_' + s + '.json'
			json_filePath = os.path.join(self.run_dir, json_file_name)
			print (json_filePath)
			json_string = json.dumps(exp_info)
			with open(json_filePath, 'w') as outfile:
				outfile.write(json_string)

	def inference_dir(self,img_dir=None):
		if img_dir == None:
			raise 'No valid img dir'

		for img_name in os.listdir(img_dir):
			a = datetime.datetime.now()
			img_path = os.path.join(img_dir,img_name)
			self.inference(img_path)
			b = datetime.datetime.now()
			# print('it takes ' + str(b-a) + ' to inference ONE image')
	
	def evaluate(self,runs_file=None, annotation_dir=None,if_vis=True):
		# load runs_file
		if runs_file==None:
			raise 'You are doing nothing, fck off'

		if annotation_dir==None and if_vis==False:
			raise 'You are doing nothing, fck off'

		# create vis dir
		self.vis_dir = os.path.join('./THzCore/Exp',self.timestring,'vis')
		if not os.path.exists(self.vis_dir):
			os.makedirs(self.vis_dir)

		# draw bounding box
		out_img_name = runs_file.split('/')[-1].replace('.json','.jpg')
		output_path = os.path.join(self.vis_dir,out_img_name)
		BoxesfromJson(runs_file,output_path,annotation_dir)

	def evaluate_dir(self,runs_dir=None, annotation_dir=None):

		for runs_filename in os.listdir(runs_dir):
			try:
				runs_file = os.path.join(runs_dir,runs_filename)
				self.evaluate(runs_file,annotation_dir)
			except:
				pass
	

if __name__ == "__main__":
	normal_folder =  './datasets/full_body/train/good'
	time_string = genTimeStamp()
	print (time_string)
	# gen config
	# config_dir = genConfig().genMultiConfig(config_idea='human_depended',normal_img_folder='./datasets/full_body/train/good')
	config_dir = genConfig(time_string).genMultiConfig(config_idea='shuffle_batch',normal_img_folder='./datasets/full_body/train/good')
	# # training
	mycore = MultiCores(mode='train',training_img_folder='./datasets/full_body/train/good',timestring=time_string)
	mycore.train_multicores(config_dir)

	# inference
	mycore = MultiCores(mode='inference',multicore_dir='./THzCore/Exp/' + time_string + '/models', timestring=time_string)
	mycore.inference_dir(img_dir='./datasets/full_body/test/objs')

	# visualize
	mycore = MultiCores(timestring=time_string)
	mycore.evaluate_dir('./THzCore/Exp/' + time_string + '/runs', annotation_dir='./datasets/full_body/Annotations')