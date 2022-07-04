from ran_utils import *
import json


class GenConfigureFiles():
    def __init__(self, 
                training_imgs_folder=None,
                save_exp_dir=None,
                save_name='config',
                config_folder=None) :
        
        self.training_imgs_folder = training_imgs_folder
        self.save_exp_dir = save_exp_dir

        # init config dir
        if config_folder == None:
            self.config_dir = os.path.join(self.save_exp_dir,genTimeStamp(),save_name)
        else:
            self.config_dir = os.path.join(config_folder,save_name)
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

    def randomGenbyNumbers(self, bacth_nums = 10,
                            num_imgs_list = [30, 50, 70, 90, 110]):
        group_imgs = {}
        for num_imgs in num_imgs_list:
            for idx in range(bacth_nums):
                key = 'Bacth_{}_NumImgs_{}'.format(idx, num_imgs)
                choose_files = random.choices(os.listdir(self.training_imgs_folder), k=num_imgs)
                selected_files =[]
                for s in choose_files:
                    if s.split('.')[-1] in IMG_FORMATS:
                        selected_files.append(s)
                group_imgs[key] = selected_files
        return group_imgs

    def genConfigFiles(self, scale=1, method='person', bacth_nums=10, 
                        num_imgs_list= [30, 50, 70, 90, 110]):
        # group_imgs = {
        #          key: [img_ids]    
        # }

        group_imgs ={}
        if method == 'person':
            group_imgs = self.genConfigByPersons()
        
        if method == 'random':
            group_imgs = self.randomGenbyNumbers(bacth_nums, num_imgs_list)
            
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
    
    def genMultiScaleFiles(self, method, bacth_nums=10, 
                        num_imgs_list= [30, 50, 70, 90, 110],
                        scale_list=[0.5,0.6,0.7,0.8,0.9,1.0]):
        for s in scale_list:
            self.genConfigFiles(scale=s, method=method, bacth_nums=bacth_nums, num_imgs_list=num_imgs_list)
        return self.config_dir


# unit test
if __name__ == "__main__":
    print ('test config generation')
    ### generate configure files
    gen_fig = GenConfigureFiles('./datasets/full_body/train/good', './TwoStageCore/exp')
    config_dir = gen_fig.genMultiScaleFiles(method='random', bacth_nums=3, 
                                            num_imgs_list = [70],
                                            scale_list=[0.1])
    print (config_dir)


