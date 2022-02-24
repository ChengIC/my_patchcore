

import os
import random
import json
from PIL import Image
import string
import random
from save_utils import genTimeStamp

def unique_id(size):
    chars = list(set(string.ascii_uppercase + string.digits).difference('LIO01'))
    return ''.join(random.choices(chars, k=size))

random.seed(20220214)

def fixTestID(abnormal_imgs_folder,div_num=11):
    image_ids = os.listdir(abnormal_imgs_folder)
    random.shuffle(image_ids)
    part_len = len(image_ids)/div_num
    fix_test_imgs_ids = image_ids[0:int(part_len)]
    usable_img_ids = image_ids[int(part_len):]
    return fix_test_imgs_ids,usable_img_ids


class GenConfig():

    def __init__(self,config_dir,
                normal_imgs_folder,abnormal_imgs_folder,
                TimeStamp=None
                ):

        self.normal_imgs_folder=normal_imgs_folder
        
        self.imgsz=None
        if self.imgsz is None:
            for im_file in os.listdir(self.normal_imgs_folder):
                if 'jpg' in im_file:
                    im_path = os.path.join(self.normal_imgs_folder,im_file)
                    im = Image.open(im_path)
                    self.imgsz = im.size[0], im.size[1]
                    continue

        self.abnormal_imgs_folder=abnormal_imgs_folder
        if TimeStamp==None:
            self.TimeStamp=genTimeStamp()
        else:
            self.TimeStamp=TimeStamp

        self.semi_supervised_config_folder = os.path.join(config_dir,self.TimeStamp,'semi')

        if not os.path.exists(self.semi_supervised_config_folder):
            os.makedirs (self.semi_supervised_config_folder)
        
        self.test_imgs_ids,self.train_img_ids = fixTestID(abnormal_imgs_folder,div_num=11)
    
    def genSemiConfig(self,
                    investigate_variables={},
                    p_factors=[1],
                    s_factors=[1],
                    sm_factors=[0],
                    maximum_percentage=20):

        for var in investigate_variables:
            if var=='percentage':
                p_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            if var=='scaling_factor':
                s_factors = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            if var=='smooth':
                sm_factors=[0, 3, 5, 7, 9, 11, 13, 15, 17]

        image_ids = os.listdir(self.normal_imgs_folder)
        random.shuffle(image_ids)
        config_path_list = []
        for p in p_factors:
            for s in s_factors:
                for sm in sm_factors:
                    train_img_ids = image_ids[0:int(p*len(image_ids)*maximum_percentage/100)]
                    data_dict = {
                        'original_imgsz':self.imgsz,
                        'percentage':float(p*maximum_percentage),
                        'scaling_factor':s,
                        'smooth':sm,
                        'test_ids':self.test_imgs_ids,
                        'train_ids':train_img_ids,
                    }
                    
                    json_file_id = unique_id(8)
                    json_filename = 'percentage_%s_smooth_%s_scaling_%s_id%s' %(str(data_dict['percentage']),
                                                                                str(data_dict['smooth']),
                                                                                str(data_dict['scaling_factor']),
                                                                                json_file_id
                                                                                )
                    json_filePath = os.path.join(self.semi_supervised_config_folder, json_filename + '.json')
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
    # Config0.genSemiConfig(investigate_variables={'percentage','scaling_factor','smooth'})
    # Config0.genSemiConfig(investigate_variables={'scaling_factor','smooth'})
    config_path_list = Config0.genSemiConfig(investigate_variables={'smooth'})
    print (config_path_list)