from gen_configV2 import GenConfig
from train_patchcore import train_patchcore
from save_utils import genTimeStamp



if __name__ == "__main__":

    config_dir='./config'
    normal_imgs_folder='./datasets/THz_Body/train/good'
    objs_imgs_folder='./datasets/full_body/test/objs'

    TimeStamp = genTimeStamp()

    config0=GenConfig(config_dir=config_dir,
                      normal_imgs_folder=normal_imgs_folder,
                      abnormal_imgs_folder=objs_imgs_folder,
                      TimeStamp=TimeStamp
                    )   
    config_path_list = config0.genSemiConfig(investigate_variables={'percentage','smooth','scaling_factor'}, maximum_percentage=2)
    

    for configPath in config_path_list:
        print ('Training with config' + configPath)
        try:
            my_training = train_patchcore(configPath=configPath,
                                        train_imgs_folder=normal_imgs_folder,
                                        resize=None,
                                        center_crop=None,
                                        f_coreset=.20,
                                        backbone_name="wide_resnet50_2",
                                        TimeStamp=TimeStamp,
                                        )
            my_training.run()
        except:
            pass

