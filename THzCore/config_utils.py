
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

random.seed(20220329)

class genConfig():

    def __init__(self, timestring=None):
        if timestring==None:
            self.config_dir = os.path.join('./THzCore/Exp',genTimeStamp(),'config')
        else:
            self.config_dir = os.path.join('./THzCore/Exp',timestring,'config')
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)

    def genMultiConfig(self,config_idea='default',
                normal_img_folder=None):

        set_scale=0.5
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
                    'scale':set_scale,
                }
                config_data['info'] = "image scale: {}, config idea: {}, human model: {} ".format(config_data['scale'],config_idea,human_model )
                
                # save config data
                json_file_name = config_data['config_id']+'.json'
                json_filePath = os.path.join(self.config_dir, json_file_name)
                json_string = json.dumps(config_data)
                with open(json_filePath, 'w') as outfile:
                    outfile.write(json_string)

        elif config_idea == 'shuffle_batch':
            file_list = os.listdir(normal_img_folder)
            random.shuffle(file_list)
            img_ids = []
            idx=0
            for img_file in file_list:
                img_ids.append(img_file)
                if len(img_ids)>= int(0.1*len(file_list)):
                    config_data = {
                        'img_ids':img_ids,
                        'config_id':'bacth_' + str(idx) + '_' + unique_id(8),
                        'scale':set_scale,
                        }
                    config_data['info'] = "image scale: {}, config idea: {}, bacth idx: {} ".format(config_data['scale'],config_idea,idx)

                    # save config data
                    json_file_name = config_data['config_id']+'.json'
                    json_filePath = os.path.join(self.config_dir, json_file_name)
                    json_string = json.dumps(config_data)
                    with open(json_filePath, 'w') as outfile:
                        outfile.write(json_string)

                    # reset image id and batch idx
                    img_ids = []
                    idx+=1
        else:
            pass

        print ('Finish configs generation')
        return self.config_dir

    def bagging_config(self,num_sets=10,size_of_subset=None,normal_img_folder=None,
                        bootstrap=True,set_scale=0.5):

        file_list = os.listdir(normal_img_folder)
        if size_of_subset==None:
            size_of_subset = int(len(file_list)/num_sets)
        if normal_img_folder:
            if bootstrap:
                for idx in range(num_sets):
                    img_ids = random.sample(file_list,size_of_subset)
                    config_data = {
                        'img_ids':img_ids,
                        'config_id':'bootstrap_bacth_' + str(idx) + '_' + unique_id(8),
                        'scale':set_scale,
                    }
                    config_data['info'] = "image scale: {}, config idea: {}, bacth idx: {} num_sets: {}".format(config_data['scale'],'bootstrap',idx,num_sets)

                    # save config data
                    json_file_name = config_data['config_id']+'.json'
                    json_filePath = os.path.join(self.config_dir, json_file_name)
                    json_string = json.dumps(config_data)
                    with open(json_filePath, 'w') as outfile:
                        outfile.write(json_string)

            else:
                # no bootstrap equlats to non disjoint sampling
                random.shuffle(file_list)
                for idx in range(0,len(file_list),size_of_subset):
                    img_ids = file_list[idx:idx+size_of_subset]
                    if len(img_ids)<size_of_subset:
                        continue
                    config_data = {
                        'img_ids':img_ids,
                        'config_id':'disjoint_bacth_' + str(int(idx/size_of_subset)) + '_' + unique_id(8),
                        'scale':set_scale,
                    }
                    config_data['info'] = "image scale: {}, config idea: {}, bacth idx: {} num_sets: {}".format(config_data['scale'],'disjoint',int(idx/size_of_subset),num_sets)

                    # save config data
                    json_file_name = config_data['config_id']+'.json'
                    json_filePath = os.path.join(self.config_dir, json_file_name)
                    json_string = json.dumps(config_data)
                    with open(json_filePath, 'w') as outfile:
                        outfile.write(json_string)
        else:
            raise 'No image folder for training'

        return self.config_dir

    def multiscale_config(self,
                        bags_num=20,
                        max_imgs_per_bag=100,
                        scale_list = [0.5,0.6,0.7,0.8,0.9,1.0],
                        normal_img_folder=None,):
        file_list = os.listdir(normal_img_folder)
        imgs_this_bag = max_imgs_per_bag
        for s in scale_list:
            imgs_this_bag = int(imgs_this_bag/2)+10
            # print (imgs_this_bag)
            for idx in range(bags_num):
                img_ids = random.sample(file_list,imgs_this_bag)
                config_data = {
                    'img_ids':img_ids,
                    'config_id':'bootstrap_bacth_scale_' + str(s) + '_'+ str(idx) + '_' + unique_id(8),
                    'scale':s,
                }
                config_data['info'] = "image scale: {}, config idea: {}, imgs num: {}".format(config_data['scale'],'multi_scale_bootstrap',imgs_this_bag)

                # save config data
                json_file_name = config_data['config_id']+'.json'
                json_filePath = os.path.join(self.config_dir, json_file_name)
                json_string = json.dumps(config_data)
                with open(json_filePath, 'w') as outfile:
                    outfile.write(json_string)
        
        return self.config_dir


    def fix_imgs_multiscale(self,
                            max_imgs_per_bag=100,
                            normal_img_folder=None):
        file_list = os.listdir(normal_img_folder)
        imgs_this_bag = max_imgs_per_bag
        img_ids = random.sample(file_list,imgs_this_bag)
        scale_list = [i/100 for i in range(50,105,5)]
        for s in scale_list:
            config_data = {
                    'img_ids':img_ids,
                    'config_id':'bootstrap_bacth_scale_' + str(s) + '_' + unique_id(8),
                    'scale':s,
                }
            config_data['info'] = "image scale: {}, config idea: {}, imgs num: {}".format(config_data['scale'],'multi_scale_bootstrap',imgs_this_bag)

            # save config data
            json_file_name = config_data['config_id']+'.json'
            json_filePath = os.path.join(self.config_dir, json_file_name)
            json_string = json.dumps(config_data)
            with open(json_filePath, 'w') as outfile:
                outfile.write(json_string)

        return self.config_dir





# if __name__ == "__main__":
#     config = genConfig()
# #   config_dir = config.bagging_config(normal_img_folder='./datasets/full_body/train/good',bootstrap=True)
#     config_dir = genConfig().multiscale_config(
#                         bags_num=20,
#                         max_imgs_per_bag=400,
#                         scale_list = [0.5,0.6,0.7,0.8,0.9,1.0],
#                         normal_img_folder='./datasets/full_body/train/good')


    