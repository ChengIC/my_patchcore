
from train_utils import *
from inference_utils import *
from ran_utils import *
import random
import json
from tqdm import tqdm
from visualize_utils import *
import shutil

random.seed(19940308)

# generate two configurations front and back
def genConfigFile(exp_dir, img_dir, scale=1, info='front', num_of_imgs=10):

    config_dir = os.path.join(exp_dir, 'config_' + info)
   
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    qualified_list = []
    for img_file in os.listdir(img_dir):
        if info in img_file:
            qualified_list.append(img_file)
    random.shuffle(qualified_list)
    selected_list = random.choices(qualified_list, k=num_of_imgs)

    config_data = {}
    config_data['filename'] = '{}_{}.json'.format(unique_id(12), info)
    config_data['img_folder'] = img_dir
    config_data['img_ids'] = selected_list
    config_data['scale'] = scale

    json_string = json.dumps(config_data)
    json_file_path = os.path.join(config_dir, config_data['filename'])
    with open(json_file_path, 'w') as outfile:
        outfile.write(json_string)

    return config_dir


# unit test
if __name__ == "__main__":
    scale = 0.1
    models_num = 1
    for num in [2]:
        exp_dir = './FrontBackCore/exp/scale{}_num{}_models{}_'.format(scale, num, models_num) + genTimeStamp()
        img_dir = './datasets/full_body/train/good'

        ############
        ##### Generate Config Files
        ############
        config_dir1, config_dir2 = None, None
        for i in range(models_num):
            config_dir1 = genConfigFile(exp_dir, img_dir, scale=scale, info='front', num_of_imgs=num)
            config_dir2 = genConfigFile(exp_dir, img_dir, scale=scale, info='back', num_of_imgs=num)

        ############
        ##### Training Model
        ############
        for config_dir in [config_dir1, config_dir2]:
            model_dir = TrainPatchCore(config_dir).trainModel()
            print ('finished training model of {}'.format(model_dir))


        # ############
        # ##### inference  
        # ############
        obj_dir = './datasets/full_body/test/objs'
        img_files = os.listdir(obj_dir)[0:20]

        for single_model_dir in os.listdir(model_dir):
                
            model_path = os.path.join(model_dir, single_model_dir)

            if os.path.isdir(model_path):
                print ('loading model from {}'.format(model_path))
                run_core = InferenceCore(model_path)

                for img_file in tqdm(img_files):
                    result, single_run_dir = run_core.inference_one_img(os.path.join(obj_dir,img_file))

                    json_string = json.dumps(result)
                    json_filename = '{}_by_{}.json'.format(img_file.split('.jpg')[0],single_model_dir)
                    json_file_path = os.path.join(single_run_dir, json_filename)
                    with open(json_file_path, 'w') as outfile:
                        outfile.write(json_string)

        # ############
        # ##### Visulization  
        # ############
        all_run_dir = '/'.join(single_run_dir.split('/')[:-1])
        print ('The overall runs dir is {}'.format(all_run_dir))
        for run_dir in os.listdir(all_run_dir):
            single_run_path = os.path.join(all_run_dir,run_dir)
            vis = VisRuns(single_run_path)
            vis.vis_all_runs(img_dir= './datasets/full_body/test/objs',
                            annotation_dir='./datasets/full_body/Annotations')
        

        #####
        # remove models to save storage
        shutil.rmtree(model_dir)
                