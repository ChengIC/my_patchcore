
import json
import random
from my_utils.ran_utils import *




# generate configure files
def genConfigFile(config_dir, img_dir, scale=1, info='front', num_of_imgs=10, seed_num=0):
    random.seed(seed_num)
    qualified_list = []
    for img_file in os.listdir(img_dir):
        if info in img_file:
            qualified_list.append(img_file)

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