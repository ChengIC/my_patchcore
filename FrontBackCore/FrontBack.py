
from train_utils import *
from inference_utils import *
from ran_utils import *
import random
import json
from tqdm import tqdm

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
    exp_dir = './FrontBackCore/exp/' + genTimeStamp()
    img_dir = './datasets/full_body/train/good'

    ############
    ##### training 
    ############
    config_dir1, config_dir2 = None, None
    for i in range(3):
        config_dir1 = genConfigFile(exp_dir, img_dir, scale=1, info='front', num_of_imgs=3)
        config_dir2 = genConfigFile(exp_dir, img_dir, scale=1, info='back', num_of_imgs=3)
    
    for config_dir in [config_dir1, config_dir2]:
        model_dir = TrainPatchCore(config_dir).trainModel()
        print ('finished training model of {}'.format(model_dir))


    ############
    ##### inference  
    ############
    obj_dir = './datasets/full_body/test/objs'
    ### model_dir = './FrontBackCore/exp/2022_07_06_16_42_45/models'
    for single_model_dir in os.listdir(model_dir):
        model_path = os.path.join(model_dir, single_model_dir)
        info = single_model_dir.split('_')[-1]
        
        # if os.path.isdir(model_path):
        #     print ('loading model from {}'.format(model_path))
        #     run_core = InferenceCore(model_path)
        #     # inference img dir
        #     for img_file in tqdm(os.listdir(obj_dir)):
        #         if info in img_file:
        #             result = run_core.inference_one_img(os.path.join(obj_dir,img_file))
        #             print (result)
