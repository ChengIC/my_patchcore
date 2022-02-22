import os
from test_patchcore import RunPatchcore


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

