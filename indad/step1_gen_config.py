

import os
import random
import json
from PIL import Image
import string
import random

def unique_id(size):
    chars = list(set(string.ascii_uppercase + string.digits).difference('LIO01'))
    return ''.join(random.choices(chars, k=size))

random.seed(20220214)

class GenConfig():
    # set the config folders
    def __init__(self,config_dir,normal_imgs_folder,objs_imgs_folder):

        self.normal_imgs_folder=normal_imgs_folder
        self.objs_imgs_folder=objs_imgs_folder

        self.supervised_config_folder = os.path.join(config_dir,'supervised')
        self.semi_supervised_config_folder = os.path.join(config_dir,'semi')

        if not os.path.exists(self.supervised_config_folder):
            os.makedirs (self.supervised_config_folder)
        if not os.path.exists(self.semi_supervised_config_folder):
            os.makedirs (self.semi_supervised_config_folder)

    def genSupervisedConfig(self, batch_size = 16, epochs = 600, imgsz = 640):

        image_ids = os.listdir(self.objs_imgs_folder)
        random.shuffle(image_ids)
        part_len = len(image_ids)/11
        self.test_imgs_ids = image_ids[0:int(part_len)]

        ratios = [2,3,4,5,6,7,8,9,10]
        for r in ratios:
            val_img_ids = []
            train_img_ids = image_ids[int(part_len):int((r+1)*part_len)]
            for i in image_ids:
                if i not in train_img_ids and i not in self.test_imgs_ids:
                    val_img_ids.append(i)

            data_dict = {
                        'batch_size':batch_size,
                        'epochs':epochs,
                        'imgsz':imgsz,
                        'ratio':r,
                        'test_ids':self.test_imgs_ids ,
                        'train_ids':train_img_ids,
                        'val_ids':val_img_ids
            }
            json_string = json.dumps(data_dict)
            json_file_id = unique_id(8)
            json_filePath = os.path.join(self.supervised_config_folder, 'ratio_' + str(r) + '_' + json_file_id + '.json')
            with open(json_filePath, 'w') as outfile:
                outfile.write(json_string)


    def genSemiConfig(self,imgsz=None,percentage=100):
        config_path_list = []
        if imgsz is None:
            for im_file in os.listdir(self.normal_imgs_folder):
                if 'jpg' in im_file:
                    im_path = os.path.join(self.normal_imgs_folder,im_file)
                    im = Image.open(im_path)
                    imgsz = im.size[0], im.size[1]
                    continue

        image_ids = os.listdir(self.normal_imgs_folder)
        random.shuffle(image_ids)
        ratios = [0.1, 0.2, 0.3, 0.4,
                    0.5, 0.6, 0.7, 0.8, 0.9, 1]
        for r in ratios:
            train_img_ids = image_ids[0:int(r*len(image_ids)*percentage/100)]
            val_img_ids = []
            for i in image_ids:
                if i not in train_img_ids and i not in self.test_imgs_ids:
                    val_img_ids.append(i)
            data_dict = {
                        'imgsz':imgsz,
                        'percentage':int(r*percentage),
                        'test_ids':self.test_imgs_ids,
                        'train_ids':train_img_ids,
                        'val_ids':val_img_ids
            }
            json_file_id = unique_id(8)
            json_filePath = os.path.join(self.semi_supervised_config_folder,'percentage_'+str((r*percentage))+ '_' + json_file_id + '.json')
            json_string = json.dumps(data_dict)
            with open(json_filePath, 'w') as outfile:
                outfile.write(json_string)
            
            config_path_list.append(json_filePath)
        return config_path_list


if __name__ == "__main__":
    config_dir = './config'
    normal_imgs_folder = './datasets/THz_Body/train/good'
    objs_imgs_folder = './datasets/full_body/test/objs'
    Config0 = GenConfig (config_dir,normal_imgs_folder,objs_imgs_folder)
    Config0.genSupervisedConfig()
    config_path_list = Config0.genSemiConfig(percentage=2)
    print (config_path_list)



