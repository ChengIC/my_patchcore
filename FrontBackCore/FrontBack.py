
from train_utils import *
from inference_utils import *
from ran_utils import *
import random
import json
from tqdm import tqdm
from visualize_utils import *

# random.seed(19940308)

# generate two configurations front and back

def genConfigFile(exp_dir, img_dir, scale=1, info='front', num_of_imgs=10):

    config_dir = os.path.join(exp_dir, 'config_' + info)
   
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    qualified_list = []
    for img_file in os.listdir(img_dir):
        if info in img_file:
            qualified_list.append(img_file)
    random.shuffle(qualified_list)
    selected_list = random.choices(qualified_list, k=num_of_imgs)

    config_data = {}
    config_data['filename'] = '{}_{}.json'.format(unique_id(12), info)
    config_data['img_folder'] = img_dir
    config_data['img_ids'] = selected_list
    config_data['scale'] = scale

    json_string = json.dumps(config_data)
    json_file_path = os.path.join(config_dir, config_data['filename'])
    with open(json_file_path, 'w') as outfile:
        outfile.write(json_string)

    return config_dir



# unit test
if __name__ == "__main__":
    exp_dir = './FrontBackCore/exp/scale1_num30_models30_' + genTimeStamp()
    img_dir = './datasets/full_body/train/good'

    ############
    ##### Generate Config Files
    ############
    config_dir1, config_dir2 = None, None
    for i in range(2):
        random.seed(i)
        config_dir1 = genConfigFile(exp_dir, img_dir, scale=0.1, info='front', num_of_imgs=2)
        config_dir2 = genConfigFile(exp_dir, img_dir, scale=0.1, info='back', num_of_imgs=2)

    ############
    ##### Training Model
    ############
    for config_dir in [config_dir1, config_dir2]:
        model_dir = TrainPatchCore(config_dir).trainModel()
        print ('finished training model of {}'.format(model_dir))


    # ############
    # ##### inference  
    # ############
    obj_dir = './datasets/full_body/test/objs'
    
    for single_model_dir in os.listdir(model_dir):
        
        model_path = os.path.join(model_dir, single_model_dir)

        if os.path.isdir(model_path):
            print ('loading model from {}'.format(model_path))
            run_core = InferenceCore(model_path)
            for img_file in tqdm(os.listdir(obj_dir)):
                result, run_dir = run_core.inference_one_img(os.path.join(obj_dir,img_file))
                json_string = json.dumps(result)
                json_filename = '{}_by_{}.json'.format(img_file.split('.jpg')[0],single_model_dir)
                json_file_path = os.path.join(run_dir, json_filename)
                with open(json_file_path, 'w') as outfile:
                    outfile.write(json_string)

            vis = VisRuns(runs_dir=run_dir)
            vis.vis_all_runs(img_dir= './datasets/full_body/test/objs',
                            annotation_dir='./datasets/full_body/Annotations')
            
                