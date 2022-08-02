import os
from my_utils.inference_utils import *
from my_utils.vis_utils import *


def self_test(model_path, percentage=0.9):
    print ('loading model from {}'.format(model_path))
    run_core = InferenceCore(model_path)

    print ('load training images')
    loader_info_json_path = os.path.join(model_path, 'loader_info.json')
    with open(loader_info_json_path) as json_file: info_data = json.load(json_file)
    img_folder = info_data['img_folder']
    img_files = info_data['img_ids']

    print ('inference training images to see self-test results')
    val_lsit = []
    all_results_score = {}
    for img_file in tqdm(img_files):
        result = run_core.inference_one_img(os.path.join(img_folder,img_file))
        all_results_score [img_file] = result['count_pixels'][0.5]
        val_lsit.append(result['count_pixels'][0.5])
    
    print ('filter inqualified image ids')
    good_ids = []
    outlier  = np.quantile(val_lsit,percentage)
    for k in all_results_score:
        if all_results_score[k]< outlier:
            good_ids.append(k)

    return good_ids