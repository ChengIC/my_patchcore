  
from gen_config import GenConfig
from train_patchcore import train_patchcore
from test_patchcore import RunPatchcore
from visual_patchcore import agvPatchCore



if __name__ == "__main__":

    config_dir = './config'
    normal_imgs_folder = './datasets/THz_Body/train/good'
    objs_imgs_folder = './datasets/full_body/test/objs'
    Config0 = GenConfig (config_dir,normal_imgs_folder,objs_imgs_folder)
    Config0.genSupervisedConfig()
    config_path_list = Config0.genSemiConfig(percentage=100)
    
    
    
    print (config_path_list)
