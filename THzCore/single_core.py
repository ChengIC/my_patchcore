
from cores_utils import *
from train_utils import *
from models import PatchCore
import warnings # for some torch warnings regarding depreciation
warnings.filterwarnings("ignore")
from tqdm import tqdm

class single_core():

	def __init__(self,mode='train',model_dir=None,
				exp_dir='./THzCore/Exp/',timestring=None):

		self.mode=mode
		
		self.timestring = timestring
		if self.timestring==None:
			self.timestring = genTimeStamp()
		self.exp_dir = exp_dir + self.timestring
		if not os.path.exists(self.exp_dir):
			os.makedirs(self.exp_dir)
		
		if self.mode == 'inference':
			self.model_dir = model_dir
			# load model
			self.model = PatchCore(f_coreset=.10, 
								backbone_name="wide_resnet50_2")
			self.model, self.model_paras, self.resize_box = load_model_from_dir(self.model, self.model_dir)
		

	def train(self,train_imgs_folder,
				config_path=None,
				default_percentgae=0.1,default_scale=0.5):

		if config_path!= None:
			# load config
			with open(config_path) as json_file:
				config_data = json.load(json_file)

			# generate training dataset
			DS = genDS(training_folder=train_imgs_folder,
						scale=config_data['scale'],
						resize_box=None)
			train_ds, _, ds_info = DS.genTrainDS_by_config(config_data)

			# create saved model_dir 
			self.model_dir = os.path.join(self.exp_dir,'models', config_data['config_id'])
			if not os.path.exists(self.model_dir):
				os.makedirs(self.model_dir)

		else:
			# generate training dataset
			DS = genDS(training_folder=train_imgs_folder,
						scale=default_scale,
						resize_box=None)
			train_ds, _, ds_info = DS.genTrainDS(default_percentgae)
			
			# create saved model_dir 
			self.model_dir = os.path.join(self.exp_dir,'models','percentage{}_scale{}'.format(default_percentgae,default_scale))
			if not os.path.exists(self.model_dir):
				os.makedirs(self.model_dir)

		# train model 
		self.model = PatchCore(f_coreset=.10, 
								backbone_name="wide_resnet50_2")
		self.tobesaved = self.model.fit(train_ds)
		
		# save model
		train_tar = os.path.join(self.model_dir,'Core_Path.tar')
		train_path = os.path.join(self.model_dir,'Core_Path')
		torch.save(self.tobesaved, train_tar)
		torch.save(self.model.state_dict(), train_path)

		# save data loader for inference
		json_filePath = os.path.join(self.model_dir, 'loader_info.json')
		json_string = json.dumps(ds_info)
		with open(json_filePath, 'w') as outfile:
			outfile.write(json_string)
		
		print('Finish training return saved model dir')
		return self.model_dir


	def inference(self,img_path):
		# create exp dir
		self.run_dir = os.path.join(self.exp_dir,'runs')
		if not os.path.exists(self.run_dir):
			os.makedirs(self.run_dir)
		
		# # load model
		# model = PatchCore(f_coreset=.10, 
		# 					backbone_name="wide_resnet50_2")
		# model, model_paras, resize_box = load_model_from_dir(model, self.model_dir)

		# load image tensor
		loader = loader_from_resize(self.resize_box)
		image = Image.open(img_path).convert('RGB')
		original_size_width, original_size_height = image.size
		image = loader(image).unsqueeze(0)
		test_img_tensor = image.to('cpu', torch.float)

		# inference
		HeatMap_Size = [original_size_height, original_size_width]
		_, pxl_lvl_anom_score = self.model.inference (test_img_tensor, self.model_paras, HeatMap_Size)
		detected_box_list = PixelScore2Boxes(pxl_lvl_anom_score)

		# log exp
		img_id = img_path.split('/')[-1].split('.')[0]
		s = self.model_dir.split('/')[-1]
		json_file_name = img_id + '_config_' + s + '.json'
		json_filePath = os.path.join(self.run_dir, json_file_name)
		
		exp_info ={
			'img_path':img_path,
			'detected_box_list':detected_box_list,
			'resize_box':self.resize_box,
			'model_dir':self.model_dir,
			'img_id':img_id,
			'json_id':s
		}

		json_string = json.dumps(exp_info)
		with open(json_filePath, 'w') as outfile:
			outfile.write(json_string)

	def inference_dir(self,img_dir=None):
		if img_dir == None:
			raise 'No valid img dir'

		for img_name in tqdm(os.listdir(img_dir)):
			img_path = os.path.join(img_dir,img_name)
			self.inference(img_path)

if __name__ == "__main__":
	normal_folder =  './datasets/full_body/train/good'
	time_string = genTimeStamp()
	print ('Start experiment at {}'.format(time_string))

	#### training
	mycore = single_core(mode='train',timestring=time_string)
	saved_model_dir = mycore.train(normal_folder,default_percentgae=0.8, default_scale=0.5) # test scale = 0.1

	#### inference
	mycore = single_core(mode='inference',model_dir=saved_model_dir,timestring=time_string)
	mycore.inference_dir(img_dir='./datasets/full_body/test/objs')
