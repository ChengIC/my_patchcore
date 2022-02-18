  
from gen_config import GenConfig
from train_patchcore import train_patchcore
from save_utils import genTimeStamp


if __name__ == "__main__":
    log_data = {}
    
    config_dir='./config'
    normal_imgs_folder='./datasets/THz_Body/train/good'
    objs_imgs_folder='./datasets/full_body/test/objs'

    config0=GenConfig(config_dir=config_dir,
                      normal_imgs_folder=normal_imgs_folder,
                      objs_imgs_folder=objs_imgs_folder
                    )   
    config_path_list = config0.genSemiConfig(percentage=30)
    
    TimeStamp = genTimeStamp()


    for configPath in config_path_list:
        try:
            my_training = train_patchcore(configPath=configPath,
                                        train_imgs_folder=normal_imgs_folder,
                                        resize=None,
                                        center_crop=None,
                                        scaling_factor=None,
                                        f_coreset=.20,
                                        backbone_name="wide_resnet50_2",
                                        TimeStamp=TimeStamp,
                                        )
            my_training.run()
        except:
            continue
        

