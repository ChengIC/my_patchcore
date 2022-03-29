
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


class genConfig():

    def __init__(self):

        self.config_dir = os.path.join('./THzCore','Config', genTimeStamp())
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)

    def genMultiConfig(self,config_idea='default',
                normal_img_folder=None):

        if config_idea == 'human_depended':
            imgs_dict = {}
            for img_file in os.listdir(normal_img_folder):      
                human_model = img_file.split('_')[2]
                if human_model not in imgs_dict:
                    imgs_dict[human_model]=[]
                imgs_dict[human_model].append(img_file)
                
            for human_model in imgs_dict:
                config_data = {
                    'img_ids':imgs_dict[human_model],
                    'config_id':human_model + '_' + unique_id(8),
                    'scale':0.5,
                }
                config_data['info'] = "image scale: {}, config idea: {}, human model: {} ".format(config_data['scale'],'human_depended',human_model )
                
                # save config data
                json_file_name = config_data['config_id']+'.json'
                json_filePath = os.path.join(self.config_dir, json_file_name)
                json_string = json.dumps(config_data)
                with open(json_filePath, 'w') as outfile:
                    outfile.write(json_string)
        
        print ('Finish configs generation')
        return self.config_dir

# if __name__ == "__main__":
#     config = genConfig()
#     config_dir = config.genMultiConfig(config_idea='human_depended',normal_img_folder='./datasets/full_body/train/good')



    