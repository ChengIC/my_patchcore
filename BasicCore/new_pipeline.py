import os 
import json
import time
import random
import string
from models import PatchCore
from tqdm import tqdm
import numpy as np
import torch
from torch import tensor
import os
from torch.utils.data import DataLoader,TensorDataset
from PIL import Image
from torchvision import transforms
import random
import cv2
from skimage.measure import label, regionprops
import pandas as pd

# Generate configure files
IMAGENET_MEAN = tensor([.485, .456, .406])
IMAGENET_STD = tensor([.229, .224, .225])

def tensor_to_img(x, normalize=False):
    if normalize:
        x *= IMAGENET_STD.unsqueeze(-1).unsqueeze(-1)
        x += IMAGENET_MEAN.unsqueeze(-1).unsqueeze(-1)
    x =  x.clip(0.,1.).permute(1,2,0).detach().numpy()
    return x

def pred_to_img(x, range): 
    range_min, range_max = range
    x -= range_min
    if (range_max - range_min) > 0:
        x /= (range_max - range_min)
    return tensor_to_img(x)

def PixelScore2Boxes(pxl_lvl_anom_score):
    anomo_thresholds= [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    score_range = pxl_lvl_anom_score.min(), pxl_lvl_anom_score.max()
    fmap_img = pred_to_img(pxl_lvl_anom_score, score_range)
    detected_box_list = {}
    for anomo_threshold in anomo_thresholds:
        mask = fmap_img > anomo_threshold
        label_mask = label(mask[:, :, 0])
        props = regionprops(label_mask)
        detected_box_list[str(anomo_threshold)] = []
        for prop in props:
            detected_box =  [int(prop.bbox[1]), int(prop.bbox[0]),
                            int(prop.bbox[3]), int(prop.bbox[2])]  # 1 0 3 2
            detected_box_list[str(anomo_threshold)].append(detected_box)

    return detected_box_list

def genTimeStamp():
    now = int(time.time())
    timeArray = time.localtime(now)
    TimeStamp = time.strftime("%Y_%m_%d_%H_%M_%S", timeArray)
    return TimeStamp

def unique_id(size):
    chars = list(set(string.ascii_uppercase + string.digits).difference('LIO01'))
    return ''.join(random.choices(chars, k=size))


def mean_size_folder(training_folder):
    height_list=[]
    width_list=[]
    for img_file in os.listdir(training_folder):
        if img_file.split('.')[-1] in IMG_FORMATS:
            img_path = os.path.join(training_folder,img_file)
            im = cv2.imread(img_path)
            h, w, _ = im.shape
            height_list.append(h)
            width_list.append(w)
    width_list = np.array(width_list)
    height_list = np.array(height_list)
    return [int(np.mean(height_list)),int(np.mean(width_list))]


IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']

class GenConfigureFiles():
    def __init__(self, 
                training_imgs_folder,
                save_exp_dir) :
        
        self.training_imgs_folder = training_imgs_folder
        self.save_exp_dir = save_exp_dir

        # init config dir
        self.config_dir = os.path.join(self.save_exp_dir,genTimeStamp(),'config')
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)

    def genConfigByPersons(self):
        person_imgs = {}
        for img_file in os.listdir(self.training_imgs_folder):
            if img_file.split('.')[-1] in IMG_FORMATS:
                person_id = img_file.split('_')[2]
                if person_id not in person_imgs:
                    person_imgs[person_id]=[]
                    person_imgs[person_id].append(img_file)
                else:
                    person_imgs[person_id].append(img_file)
        
        return person_imgs

    def genConfigFiles(self, scale=1, method='person'):
        # group_imgs = {
        #          key: [img_ids]    
        # }

        group_imgs ={}
        if method == 'person':
            group_imgs = self.genConfigByPersons()

        for k, v in group_imgs.items():
            config_data = {}
            config_data['filename'] = k + '_' + unique_id(8) + '.json'
            config_data['img_ids'] = v
            config_data['scale'] = scale
            config_data['img_folder'] = self.training_imgs_folder

            json_filePath = os.path.join(self.config_dir, config_data['filename'])
            json_string = json.dumps(config_data)
            with open(json_filePath, 'w') as outfile:
                outfile.write(json_string)
    
    def genMultiScaleFiles(self, method, scale_list=[0.5,0.6,0.7,0.8,0.9,1.0]):
        for s in scale_list:
            self.genConfigFiles(scale=s, method=method)
        return self.config_dir

# Training PatchCore from configure file
class TrainPatchCore ():
    def __init__(self, config_dir):
        self.config_dir = config_dir
        self.model_dir = os.path.join ('/'.join(self.config_dir.split('/')[:-1]), 'models')
        # init model dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def genDataSet(self,config_data):

        # create dataloader
        IMAGENET_MEAN = tensor([.485, .456, .406])
        IMAGENET_STD = tensor([.229, .224, .225])
        transfoms_paras = [
                            transforms.ToTensor(),
                            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                        ]
        
        mean_img_size = mean_size_folder(config_data['img_folder'])
        resize_box = [int(mean_img_size[0]*config_data['scale']),
                    int(mean_img_size[1]*config_data['scale'])]

        if config_data['scale']!=1:
            transfoms_paras.append(transforms.Resize(resize_box, interpolation=transforms.InterpolationMode.BICUBIC))

        loader = transforms.Compose(transfoms_paras)

        # create tensor dataset
        train_ims = []
        train_labels = []
        for img_id in config_data['img_ids']:
            img_path = os.path.join(config_data['img_folder'], img_id)
            train_im = loader(Image.open(img_path).convert('RGB'))
            train_label = tensor([0])

            train_ims.append(train_im.numpy())
            train_labels.append(train_label.numpy())
        
        train_ims = torch.from_numpy(np.array(train_ims))
        train_labels = torch.from_numpy(np.array(train_labels))

        print ('Training Tensor Shape is' + str(train_ims.shape))

        train_ds = DataLoader(TensorDataset(train_ims,train_labels))

        return train_ds, resize_box

    def trainModel(self):
        # load config file 
        for config_file in tqdm(os.listdir(self.config_dir)):
            if config_file.split('.')[-1] == 'json':
                with open(os.path.join(self.config_dir, config_file)) as json_file:
                    config_data = json.load(json_file)
                    train_ds, resize_box = self.genDataSet(config_data)

                    # train model 
                    model = PatchCore(f_coreset=.10, 
                                        backbone_name="wide_resnet50_2")

                    tobesaved = model.fit(train_ds)
                    
                    # save model
                    model_path = os.path.join(self.model_dir,config_file.split('.')[0])
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    train_tar = os.path.join(model_path,'Core_Path.tar')
                    train_path = os.path.join(model_path, 'Core_Path')
                    torch.save(tobesaved, train_tar)
                    torch.save(model.state_dict(), train_path)

                    # save training info
                    config_data['resize_box']=resize_box

                    json_filePath = os.path.join(model_path, 'loader_info.json')
                    json_string = json.dumps(config_data)
                    with open(json_filePath, 'w') as outfile:
                        outfile.write(json_string)

        print('finish training')
        return self.model_dir

class InferenceCore():

    def __init__(self, model_dir) :
        self.model_dir = model_dir
        self.run_dir = os.path.join ('/'.join(self.model_dir.split('/')[:-1]), 'runs')
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

    def load_model_from_dir(self, model_path):
        state_dict_path = os.path.join(model_path,'Core_Path')
        tar_path = os.path.join(model_path,'Core_Path.tar')
        configPath =os.path.join(model_path,'loader_info.json')

        # load model and config
        model = PatchCore(f_coreset=.10, backbone_name="wide_resnet50_2")
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(state_dict_path))
            model_paras = torch.load(tar_path)
        else:
            model.load_state_dict(torch.load(state_dict_path,map_location ='cpu'))
            model_paras = torch.load(tar_path,map_location ='cpu')
        with open(configPath) as json_file:
            config_data = json.load(json_file)

        print ('loading model dir: ' + model_path + ' sucessfully')

        return model, model_paras, config_data

    def load_image_to_tensor_cpu(self, img_path, config_data):

        IMAGENET_MEAN = tensor([.485, .456, .406])
        IMAGENET_STD = tensor([.229, .224, .225])
        transfoms_paras = [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]

        if config_data['scale']!=1:
            transfoms_paras.append(transforms.Resize(config_data['resize_box'], interpolation=transforms.InterpolationMode.BICUBIC))

        loader = transforms.Compose(transfoms_paras)

        image = Image.open(img_path).convert('RGB')
        original_size_width, original_size_height = image.size
        image = loader(image).unsqueeze(0)

        test_img_tensor = image.to('cpu', torch.float)
        HeatMap_Size = [original_size_height, original_size_width]

        return test_img_tensor, HeatMap_Size

    # inference patches
    def inference_one_model (self, img_dir):
        for single_model_dir in os.listdir(self.model_dir):
            model_path = os.path.join(self.model_dir, single_model_dir)

            # load model
            if os.path.isdir(model_path):
                model, model_paras, config_data = self.load_model_from_dir(model_path)

                # inference img one by one
                for img_file in tqdm(os.listdir(img_dir)):
                    if img_file.split('.')[-1] in IMG_FORMATS:
                        img_path = os.path.join(img_dir, img_file)
                        test_img_tensor, HeatMap_Size = self.load_image_to_tensor_cpu(img_path, config_data)
                        _, pxl_lvl_anom_score = model.inference (test_img_tensor, model_paras, HeatMap_Size)
                        try:
                            detected_box_list = PixelScore2Boxes(pxl_lvl_anom_score)

                            # log exp
                            img_id = img_file.split('/')[-1].split('.')[0]
                            json_file_name = img_id + '_config_' + single_model_dir + '.json'
                            json_filePath = os.path.join(self.run_dir, json_file_name)
                            
                            exp_info ={
                                'img_path':img_path,
                                'detected_box_list':detected_box_list,
                                'model_dir':single_model_dir,
                                'img_id':img_id,
                            }

                            json_string = json.dumps(exp_info)
                            with open(json_filePath, 'w') as outfile:
                                outfile.write(json_string)
                        except:
                            pass

        return self.run_dir
    
    # continues inference
    def continuous_inference(self, img_dir):
        inference_files_list = os.listdir(self.run_dir)
        for single_model_dir in os.listdir(self.model_dir):
            model_path = os.path.join(self.model_dir, single_model_dir)
            if os.path.isdir(model_path):
                model, model_paras, config_data = self.load_model_from_dir(model_path)

                for img_file in tqdm(os.listdir(img_dir)):
                    img_id = img_file.split('/')[-1].split('.')[0]
                    json_file_name = img_id + '_config_' + single_model_dir + '.json'

                    if json_file_name in inference_files_list:
                        print ('already inference')

                    else:
                        if img_file.split('.')[-1] in IMG_FORMATS:
                            img_path = os.path.join(img_dir, img_file)
                            test_img_tensor, HeatMap_Size = self.load_image_to_tensor_cpu(img_path, config_data)
                            _, pxl_lvl_anom_score = model.inference (test_img_tensor, model_paras, HeatMap_Size)
                            try:
                                detected_box_list = PixelScore2Boxes(pxl_lvl_anom_score)

                                # log exp
                                img_id = img_file.split('/')[-1].split('.')[0]
                                json_file_name = img_id + '_config_' + single_model_dir + '.json'
                                json_filePath = os.path.join(self.run_dir, json_file_name)
                                
                                exp_info ={
                                    'img_path':img_path,
                                    'detected_box_list':detected_box_list,
                                    'model_dir':single_model_dir,
                                    'img_id':img_id,
                                }

                                json_string = json.dumps(exp_info)
                                with open(json_filePath, 'w') as outfile:
                                    outfile.write(json_string)

                            except:
                                pass

                        else:
                            pass
                        
            return self.run_dir

class SummariseRuns():
    def __init__(self, run_dir) :
        self.run_dir = run_dir
        self.summary_dir = os.path.join ('/'.join(self.run_dir.split('/')[:-1]), 'summary')
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
    
    def returnPD(self):
        
        # define csv columns 
        runs_data = {
            'config_id':[],
            'ImageID':[],
            'selected_threshold':[],
            'XMin':[],
            'XMax':[],
            'YMin':[],
            'YMax':[]
        }

        # load multiple json files
        for file in tqdm(os.listdir(os.path.join(self.run_dir))):
            if '.json' in file:
                img_id = file.split('_config')[0]
                config_id = file.split('config_')[1].split('.json')[0]

                with open(os.path.join(self.run_dir,file)) as run_file:
                    run_data = json.load(run_file)
                    detected_box_list = run_data['detected_box_list']
                    for selected_th in detected_box_list:
                        detected_box = detected_box_list[selected_th]
                        for bb in detected_box:
                            runs_data['config_id'].append(config_id)
                            runs_data['ImageID'].append(img_id)
                            runs_data['selected_threshold'].append(float(selected_th))
                            runs_data['XMin'].append(bb[0])
                            runs_data['XMax'].append(bb[2])
                            runs_data['YMin'].append(bb[1])
                            runs_data['YMax'].append(bb[3])
                                
        runs_data = pd.DataFrame(runs_data)
        runs_data_path = os.path.join(self.summary_dir, 'runs' + genTimeStamp() + '.csv')
        runs_data.to_csv(runs_data_path,index=False)
        print('Summary file: ' + runs_data_path)

# # generate configure files
# gen_fig = GenConfigureFiles('./datasets/full_body/train/good', './BasicCore/exp')
# config_dir = gen_fig.genMultiScaleFiles(method='person')
# print (config_dir)

# # training 
# train_session = TrainPatchCore(config_dir)
# model_dir = train_session.trainModel()
# print (model_dir)

# inferencing
img_dir = './datasets/full_body/test/objs'
run_dir = InferenceCore(model_dir).inference_one_model(img_dir)
print (run_dir)

# summary
SummariseRuns(run_dir).returnPD()
