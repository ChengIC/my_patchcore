  
from unittest import result
from test_patchcore import RunPatchcore
from visual_patchcore import visPatchCore
import os


if __name__ == "__main__":
    test_imgs_folder = './datasets/full_body/test_reduced'
    for roots, dirs, files in os.walk('./model_zoo'):
        if 'patchcore_path.tar' in files:
            model_dir =  roots
            config_name = model_dir.split('/')[2]+'.json'
            configPath = os.path.join('./config/semi',config_name)
            run1 = RunPatchcore(model_dir,configPath=configPath)
            run1.run(test_imgs_folder,writeImage=True)

    for roots, dirs, files in os.walk('./results'):
        if 'data' in dirs:
            result_dir = [os.path.join(roots,'data')]
            annotation_folder = './datasets/full_body/Annotations'
            av1 = visPatchCore (result_dir,test_imgs_folder,annotation_folder)
            av1.vis_result()