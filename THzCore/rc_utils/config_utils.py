
import string
import random
import time
import os 
import json

def unique_id(size):
    chars = list(set(string.ascii_uppercase + string.digits).difference('LIO01'))
    return ''.join(random.choices(chars, k=size))

def genTimeStamp():
    now = int(time.time())
    timeArray = time.localtime(now)
    TimeStamp = time.strftime("%Y_%m_%d_%H_%M_%S", timeArray)
    return TimeStamp

IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']

random.seed(20220329)

class genConfig():

    def __init__(self, 
                config_parent_dir = './THzCore/Exp',
                normal_imgs_folder=None,
                timestring=None):

        self.config_parent_dir=config_parent_dir
        self.normal_imgs_folder=normal_imgs_folder
        self.timestring=timestring
        if self.timestring==None:
            self.timestring=genTimeStamp()

        # create config dir to store config files
        self.config_dir = os.path.join(self.config_parent_dir,self.timestring,'config')
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)

    def genSingleConfig(self, 
                        paras ={}):

        # paras['bag_idx']
        # paras['imgs_list']
        # paras['config_id']
        # paras['scale']

        verify_ids = []
        for img_file in paras['imgs_list']:
            if img_file.split('.')[-1] in IMG_FORMATS:
                verify_ids.append(img_file)

        config_data = { 
                        'img_ids':verify_ids,
                        'config_id':'BagIdx_' + str(paras['bag_idx']) + '_' + 'Scale_' + str(paras['scale']) + '_' + paras['config_id'],
                        'scale':paras['scale'],
                    }

        # save config data
        json_file_name = config_data['config_id']+'.json'
        json_filePath = os.path.join(self.config_dir, json_file_name)
        json_string = json.dumps(config_data)
        with open(json_filePath, 'w') as outfile:
            outfile.write(json_string)


    def genMultiConfigs(self,
                        num_batch = 10, 
                        imgs_per_bag=100,
                        scale_step_size=10,
                        bag_imgs_percentage=None, # 0.0 - 1.0
                        method='default'             
                    ):
        if bag_imgs_percentage!=None:
            selected_img_num = int(bag_imgs_percentage*len(self.normal_imgs_folder))
        else:
            selected_img_num = imgs_per_bag

        file_list = os.listdir(self.normal_imgs_folder)
        scale_list = [i/100 for i in range(50,160,scale_step_size)] # scale setting: 0.5, 0.55, 0.6, 0.65 ... 1.0

        if 1 not in scale_list:
            scale_list.append(1)

        if method == 'default':
            # perform multi-scale and multi-image bags
            for i in range(num_batch):
                imgs_list = random.sample(file_list,selected_img_num)

                for s in scale_list:
                    paras = {
                        'imgs_list':imgs_list,
                        'config_id':unique_id(8),
                        'bag_idx':i,
                        'scale':s,
                    }
                    self.genSingleConfig(paras)

        return self.config_dir


if __name__ == "__main__":
    normal_imgs_folder = './datasets/full_body/train/good'
    config_dir = genConfig(normal_imgs_folder = normal_imgs_folder).genMultiConfigs(num_batch=20,imgs_per_bag=50)



    