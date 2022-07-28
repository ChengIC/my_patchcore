import os
from inference_utils import *
from vis_utils import *
    
########################################
##### self test multi-patchcores  ######
########################################
model_dir = './FinalCore/exp/2022_07_24_15_48_09/models'
self_test_dir = '/'.join(model_dir.split('/')[0:-1]) + '/self_test'
if not os.path.exists(self_test_dir): os.makedirs(self_test_dir)

for single_model_dir in os.listdir(model_dir):
    model_path = os.path.join(model_dir, single_model_dir)
    if os.path.isdir(model_path):
        print ('loading model from {}'.format(model_path))
        run_core = InferenceCore(model_path)

        print ('load training images')
        loader_info_json_path = os.path.join(model_path, 'loader_info.json')
        with open(loader_info_json_path) as json_file: info_data = json.load(json_file)
        img_folder = info_data['img_folder']
        img_files = info_data['img_ids']

        print ('inference training images to see self-test results')
        for img_file in tqdm(img_files):
            result = run_core.inference_one_img(os.path.join(img_folder,img_file))
            fv_color_img = PixelScore2Img(result['pixel_score'])
            saved_path = os.path.join(self_test_dir, img_file.replace('jpg','png'))
            fv_color_img.save(saved_path)