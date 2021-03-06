

# from unicodedata import name
from single_core import single_core
import os
from config_utils import * 
# from joblib import Parallel, delayed
from hard_imgs import hard_imgs_ids

def train_multiScaleCores(normal_img_folder,
                    bags_num=20,
                    max_imgs_per_bag=100,
                    time_string=None):
    if time_string==None:
        time_string = genTimeStamp()

    # config_dir = genConfig(time_string).multiscale_config(
    #                     bags_num=bags_num,
    #                     max_imgs_per_bag=max_imgs_per_bag,
    #                     scale_list = [0.5,0.6,0.7,0.8,0.9,1.0],
    #                     normal_img_folder=normal_img_folder)
    
    config_dir = genConfig(time_string).fix_imgs_multiscale(
                            max_imgs_per_bag=max_imgs_per_bag,
                            normal_img_folder=normal_img_folder)

    for config_file in os.listdir(config_dir):
        config_filePath = os.path.join(config_dir,config_file)
        mycore = single_core(mode='train',timestring=time_string)
        saved_model_dir = mycore.train(normal_img_folder,config_path=config_filePath)

    saved_models_dir = '/'.join(saved_model_dir.split('/')[:-1])

    print('Finished training for all config paths. Save model dirs {}'.format(saved_models_dir))

    return saved_models_dir, time_string

def getSingleModelDir(models_dir):
    all_models_dir = []
    for roots, _, files in os.walk(models_dir):
       if 'Core_Path.tar' in files:
            all_models_dir.append(roots)
    return all_models_dir

def inferenceOneModel(mycore,img_path):
    mycore.inference(img_path)

def inferenceModel(mycore,img_dir):
    mycore.inference_dir(img_dir)

def inferenceModel_specific(mycore,id_list,img_dir):
    mycore.inferece_some_ims(id_list,img_dir)

if __name__ == "__main__":

    saved_models_dir, time_string = train_multiScaleCores(normal_img_folder='./datasets/full_body/train/good',
                                                            bags_num=20,
                                                            max_imgs_per_bag=10,
                                                            time_string=None)

    all_models_dir = getSingleModelDir(saved_models_dir)

    img_dir='./datasets/full_body/test/objs/'

    for model_dir in all_models_dir:
        my_core = single_core(mode='inference',model_dir=model_dir,timestring=time_string)
        inferenceModel(my_core,img_dir)


    # i = 0
    # while i + 2 < len(all_models_dir):
    #     batch_core_models = []
    #     for model_dir in all_models_dir[i:i+2]:
    #         batch_core_models.append(single_core(mode='inference',model_dir=model_dir,timestring=time_string))

    #     # Parallel(n_jobs=-1)(delayed(inferenceModel)(mycore,img_dir) 
    #     #                                             for mycore in batch_core_models)


    #     id_list = hard_imgs_ids
    #     Parallel(n_jobs=-1)(delayed(inferenceModel_specific)(mycore,id_list,img_dir) 
    #                                                 for mycore in batch_core_models)
    #     i = i + 2
            
