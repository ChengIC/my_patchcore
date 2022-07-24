
from ran_utils import *
import random
import json
from tqdm import tqdm
import shutil
from train_utils import *
from inference_utils import *
from vis_ensemble_features import *

# generate configure files
def genConfigFile(config_dir, img_dir, scale=1, info='front', num_of_imgs=10):
    
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



if __name__ == "__main__":
    
    ########################################
    ######### ensemble settings ############
    ########################################
    scale = 1
    num_imgs = 30
    models_num = 15
    
    ########################################
    ############### exp folders ############
    ########################################
    time_stamp = genTimeStamp()
    exp_dir = './FinalCore/exp/' + time_stamp
    config_dir = exp_dir + '/config'
    model_dir = exp_dir + '/models'
    # runs_dir = exp_dir + '/runs'
    vis_dir = exp_dir + '/vis'
    runs_dir = '/media/rc/backup/exp/{}/runs'.format(time_stamp)  # save to back up drive due to limited  

    if not os.path.exists(exp_dir): os.makedirs(exp_dir)
    if not os.path.exists(config_dir): os.makedirs(config_dir)
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    if not os.path.exists(runs_dir): os.makedirs(runs_dir)
    if not os.path.exists(vis_dir): os.makedirs(vis_dir)

    ########################################
    ##### generate config files ############
    ########################################
    img_dir = './datasets/full_body/train/good'
    for i in range(models_num):
        genConfigFile(config_dir, img_dir, scale=scale, info='front', num_of_imgs=num_imgs)
        genConfigFile(config_dir, img_dir, scale=scale, info='back', num_of_imgs=num_imgs)

    ########################################
    ##### train multi-patchcores ###########
    ########################################
    train_session = TrainPatchCore(config_dir, model_dir).trainModel()


    ########################################
    ##### runs multi-patchcores  ###########
    ########################################
    obj_dir = './datasets/full_body/test/objs'
    img_files = os.listdir(obj_dir)[0:10]

    for single_model_dir in os.listdir(model_dir):
        model_path = os.path.join(model_dir, single_model_dir)
        if os.path.isdir(model_path):
            print ('loading model from {}'.format(model_path))
            run_core = InferenceCore(model_path)

            for img_file in tqdm(img_files):
                result = run_core.inference_one_img(os.path.join(obj_dir,img_file))
                json_string = json.dumps(result)
                json_filename = '{}_by_{}.json'.format(img_file.split('.jpg')[0],single_model_dir)
                json_folder = os.path.join(runs_dir, img_file.split('.jpg')[0])
                if not os.path.exists(json_folder): os.makedirs(json_folder)
                json_file_path = os.path.join(json_folder, json_filename)
                with open(json_file_path, 'w') as outfile:
                    outfile.write(json_string)
    

    ########################################
    ########### vis avg results  ###########
    ########################################
    for img_run_dir in os.listdir(runs_dir):
        img_run_path = os.path.join(runs_dir, img_run_dir)
        if os.path.isdir(img_run_path):
            vis_exp = VisEnsembleFeature(obj_dir=obj_dir, 
                                        annotation_dir='./datasets/full_body/Annotations',
                                        img_run_path=img_run_path, 
                                        vis_dir=vis_dir)
            vis_exp.save_opt_fv()

