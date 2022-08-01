
from my_utils.ran_utils import *
import random
import json
from tqdm import tqdm
import shutil
from my_utils.train_utils import *
from my_utils.inference_utils import *
from my_utils.vis_ensemble_features import *
from my_utils.self_test_utils import *
from my_utils.median_filter_dir import *
from my_utils.config_utils import *
from my_utils.summary_utils import *

if __name__ == "__main__":
    
    ########################################
    ######### ensemble settings ############
    ########################################
    # scale = 0.1
    # num_imgs = 10
    # models_num = 2
    
    scale = 1
    num_imgs = 60
    models_num = 15

    ########################################
    ############### exp folders ############
    ########################################
    time_stamp = genTimeStamp()
    exp_dir = './MyCore/exp/' + time_stamp

    config_dir = exp_dir + '/config'
    model_dir = exp_dir + '/models'

    self_test_dir = exp_dir  + '/self_test'

    new_config_dir = exp_dir + '/new_config'
    new_model_dir = exp_dir + '/new_models'

    runs_dir = exp_dir + '/runs'
    # runs_dir = '/media/rc/backup/exp/{}/runs'.format(time_stamp)  # save to back up drive due to limited  
    vis_dir = exp_dir + '/vis'
    

    for dir in [config_dir, model_dir, new_config_dir, 
                new_model_dir, runs_dir, vis_dir]:
        if not os.path.exists(dir): 
            os.makedirs(dir)

    ########################################
    ##### Apply median filter to images#####
    ########################################
    FILTER_DEGREE = 5

    input_img_dir = './datasets/full_body/train/good'
    input_obj_dir = './datasets/full_body/test/objs'

    output_img_dir = './datasets/full_body/median_'+ str(FILTER_DEGREE) + '/train/good'
    output_obj_dir = './datasets/full_body/median_' + str(FILTER_DEGREE) + '/test/objs'
    if not os.path.exists(output_img_dir): os.makedirs(output_img_dir)
    if not os.path.exists(output_obj_dir): os.makedirs(output_obj_dir)
    
    median_dir(input_img_dir, output_img_dir, filter_degree=FILTER_DEGREE)
    median_dir(input_obj_dir, output_obj_dir, filter_degree=FILTER_DEGREE)

    img_dir = output_img_dir
    obj_dir = output_obj_dir

    ########################################
    ##### generate config files ############
    ########################################


    for i in range(models_num):
        genConfigFile(config_dir, img_dir, scale=scale, info='front', num_of_imgs=num_imgs)
        genConfigFile(config_dir, img_dir, scale=scale, info='back', num_of_imgs=num_imgs)

    ########################################
    ##### train multi-patchcores ###########
    ########################################
    train_session = TrainPatchCore(config_dir, model_dir).trainModel()

    #######################################
    ############# self-test  ##############
    ######  generate new config files #####
    #######################################
    new_img_ids = {}
    for single_model_dir in os.listdir(model_dir):
        model_path = os.path.join(model_dir, single_model_dir)
        if os.path.isdir(model_path): 
            new_img_ids[single_model_dir] = self_test(model_path)
    
    # modfiy config file 
    for config_file in os.listdir(config_dir):
        with open(os.path.join(config_dir, config_file)) as json_file:
            config_data = json.load(json_file)
            
            # replace new ids
            config_data['img_ids'] = new_img_ids[config_file.split('.')[0]]

            json_string = json.dumps(config_data)
            json_file_path = os.path.join(new_config_dir, config_data['filename'])
            with open(json_file_path, 'w') as outfile:
                outfile.write(json_string)

    ########################################
    ##### train new multi-patchcores #######
    ########################################
    train_new_session = TrainPatchCore(new_config_dir, new_model_dir).trainModel()


    ########################################
    ##### runs multi-patchcores  ###########
    ########################################
    
    img_files = os.listdir(obj_dir)

    for single_model_dir in os.listdir(new_model_dir):
        model_path = os.path.join(new_model_dir, single_model_dir)
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
    exp_summary = SummaryExp()
    for img_run_dir in os.listdir(runs_dir):
        img_run_path = os.path.join(runs_dir, img_run_dir)
        if os.path.isdir(img_run_path):
            vis_exp = VisEnsembleFeature(obj_dir=obj_dir, 
                                        annotation_dir='./datasets/full_body/Annotations',
                                        img_run_path=img_run_path, 
                                        vis_dir=vis_dir)
            fv, all_boxes, img_id = vis_exp.save_opt_fv()

            # compute and update results
            tp_score, fp_score = exp_summary.compute_score(fv, all_boxes)
            exp_summary.update_data_frame (img_id, tp_score, fp_score)

    exp_summary.write_exp_summary(exp_dir)

