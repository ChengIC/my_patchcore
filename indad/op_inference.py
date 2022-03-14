import os
from test_patchcore import RunPatchcore
from save_utils import genTimeStamp

if __name__ == "__main__":
    ts = genTimeStamp()
    test_imgs_folder = './datasets/full_body/test/objs'
    model_zoo_folder = './model_zoo/2022_03_01_14_51_00'
    for roots, dirs, files in os.walk(model_zoo_folder):
        if 'training_config.json' in files:
            
            config_path = os.path.join(roots,'training_config.json')
            run0 = RunPatchcore(model_dir=roots,
                                resize=None,
                                center_crop=None,
                                configPath=config_path,
                                TimeStamp=model_zoo_folder.split('/')[-1]
                                )

            run0.run(imgs_folder=test_imgs_folder,
                    writeImage=True)

