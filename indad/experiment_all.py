from gen_config import GenConfig
from train_patchcore import train_patchcore
from save_utils import genTimeStamp
import os
from test_patchcore import RunPatchcore
from visual_patchcore import visPatchCore


if __name__ == "__main__":
    
    config_dir='./config'
    normal_imgs_folder='./datasets/THz_Body/train/good'
    objs_imgs_folder='./datasets/full_body/test/objs'

    config0=GenConfig(config_dir=config_dir,
                      normal_imgs_folder=normal_imgs_folder,
                      objs_imgs_folder=objs_imgs_folder
                    )   
    config_path_list = config0.genSemiConfig(percentage=1)
    
    TimeStamp = genTimeStamp()

    for configPath in config_path_list:
        print ('Training with config' + configPath)
        try:
            my_training = train_patchcore(configPath=configPath,
                                        train_imgs_folder=normal_imgs_folder,
                                        resize=None,
                                        center_crop=None,
                                        scaling_factor=0.2,
                                        f_coreset=.20,
                                        backbone_name="wide_resnet50_2",
                                        TimeStamp=TimeStamp,
                                        )
            my_training.run()
        except:
            continue
    
    print ('Finish Training and start testing')
    test_imgs_folder = './datasets/full_body/test_reduced'
    annotation_folder = './datasets/full_body/Annotations'

    for roots, dirs, files in os.walk('./model_zoo/'+TimeStamp):
        if 'training_config.json' in files:
            
            config_path = os.path.join(roots,'training_config.json')
            run0 = RunPatchcore(model_dir=roots,
                                resize=None,
                                center_crop=None,
                                configPath=config_path,
                                TimeStamp=TimeStamp)

            run0.run(imgs_folder=test_imgs_folder,
                    writeImage=False)

    print ('Finish Inferencing and do visualization')
    for roots, dirs, files in os.walk('./results/'+TimeStamp):
         if dirs == ['imgs', 'data']:
            result_dirs = [os.path.join(roots,'data')]
            av1 = visPatchCore (all_json_dirs=result_dirs,
                                test_imgs_folder=test_imgs_folder,
                                annotation_folder=annotation_folder,
                                write_image=True,
                                write_result=True,
                                TimeStamp=TimeStamp)
            av1.vis_result()