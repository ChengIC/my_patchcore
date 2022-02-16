  
from gen_config import GenConfig
from train_patchcore import train_patchcore
from test_patchcore import RunPatchcore
from visual_patchcore import agvPatchCore
import json


if __name__ == "__main__":
    log_data = {}
    
    config_dir='./config'
    normal_imgs_folder='./datasets/THz_Body/train/good'
    objs_imgs_folder='./datasets/full_body/test/objs'

    config0=GenConfig(config_dir,normal_imgs_folder,objs_imgs_folder)   
    config_path_list = config0.genSemiConfig(percentage=20)
    
    for configPath in config_path_list:
        my_training = train_patchcore(configPath,normal_imgs_folder)
        train_tar, train_path = my_training.run()
        log_data[configPath] = {
            'train_tar':train_tar,
            'train_path':train_path
        }
    

    json_filePath = 'train_log.json'
    json_string = json.dumps(log_data)
    with open(json_filePath, 'w') as outfile:
        outfile.write(json_string)

