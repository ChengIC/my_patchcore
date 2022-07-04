
from ran_utils import *
import os 
import json
from models import PatchCore
from tqdm import tqdm
import torch
from torch import tensor
import os
from PIL import Image
from torchvision import transforms

class InferenceCore():

    def __init__(self, model_dir=None, save_name='runs') :
        self.model_dir = model_dir
        self.run_dir = os.path.join ('/'.join(self.model_dir.split('/')[:-1]), save_name)
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
                        img_lvl_anom_score, pxl_lvl_anom_score = model.inference (test_img_tensor, model_paras, HeatMap_Size)
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
                            'img_score':img_lvl_anom_score.tolist(),
                            # 'pixel_score':pxl_lvl_anom_score.tolist(),
                        }

                        json_string = json.dumps(exp_info)
                        with open(json_filePath, 'w') as outfile:
                            outfile.write(json_string)

        return self.run_dir
    
    # continues inference
    def continuous_inference(self, img_dir):
        inference_files_list = os.listdir(self.run_dir).copy()

        for single_model_dir in tqdm(os.listdir(self.model_dir)):
            model_path = os.path.join(self.model_dir, single_model_dir)

            if os.path.isdir(model_path):
                model, model_paras, config_data = self.load_model_from_dir(model_path)

                for img_file in tqdm(os.listdir(img_dir)):
                    img_id = img_file.split('/')[-1].split('.')[0]
                    json_file_name = img_id + '_config_' + single_model_dir + '.json'

                    if json_file_name in inference_files_list:
                        print ('already inference')

                    else:
                        print ('load model from {} to inference image {}'.format(model_path, img_file))
                        if img_file.split('.')[-1] in IMG_FORMATS:
                            img_path = os.path.join(img_dir, img_file)
                            test_img_tensor, HeatMap_Size = self.load_image_to_tensor_cpu(img_path, config_data)
                            img_lvl_anom_score, pxl_lvl_anom_score = model.inference (test_img_tensor, model_paras, HeatMap_Size)
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
                                'img_score':img_lvl_anom_score.tolist(),
                               # 'pixel_score':pxl_lvl_anom_score.tolist(),
                            }

                            json_string = json.dumps(exp_info)
                            with open(json_filePath, 'w') as outfile:
                                outfile.write(json_string)

        return self.run_dir


if __name__ == "__main__":
    print ('test image inference')
    ### generate configure files
    img_dir = './datasets/full_body/test/objs'
    model_dir = './TwoStageCore/exp/2022_07_01_10_55_40/models'
    run_dir =  InferenceCore(model_dir).inference_one_model(img_dir)
    print (run_dir)