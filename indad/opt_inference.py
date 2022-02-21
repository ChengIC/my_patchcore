from gen_config import GenConfig
from train_patchcore import train_patchcore
from save_utils import genTimeStamp
import os
from test_patchcore import RunPatchcore
from visual_patchcore import visPatchCore


if __name__ == "__main__":

    test_imgs_folder = './datasets/full_body/test/objs'
    for roots, dirs, files in os.walk('./model_test/'):
        if 'training_config.json' in files:
            
            config_path = os.path.join(roots,'training_config.json')
            run0 = RunPatchcore(model_dir=roots,
                                resize=None,
                                center_crop=None,
                                configPath=config_path,
                                TimeStamp=None)

            run0.run(imgs_folder=test_imgs_folder,
                    writeImage=True)

