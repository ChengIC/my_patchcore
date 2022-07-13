

import os 
from ran_utils import *
import json

def count_high(single_fmap):
    count = (single_fmap>=0.8).sum()
    return count

def selected_fmap (all_feature_images):
    energies = []
    for single_fmap in all_feature_images:
        energies.append(count_high(single_fmap))
        
    rank_list = [sorted(energies).index(x) for x in energies]

    final_weighted_list = []
    for r in rank_list:
        if r == 0:
            final_weighted_list.append(0.9)
        else:
            final_weighted_list.append(0.1/(len(rank_list)-1))
    # print (final_weighted_list)
    
    average_fmap = []
    for idx, single_fmap in enumerate(all_feature_images):
        if len(average_fmap)==0:
            average_fmap = final_weighted_list[idx] * single_fmap
        else:
            average_fmap += final_weighted_list[idx] * single_fmap
            
    average_fmap = average_fmap/len(all_feature_images)
    return average_fmap

class VisRuns():
    def __init__(self, runs_dir=None, save_name='vis') :
        self.runs_dir = runs_dir

        self.vis_dir = os.path.join ('/'.join(self.runs_dir.split('/')[:-1]), save_name)
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)
    
    def vis_average(self, img_dir=None, annotation_dir=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir

        json_by_id = {}
        for json_file_name in os.listdir(self.runs_dir):
            img_id = json_file_name.split('_by')[0]
            if img_id not in json_by_id:
                json_by_id[img_id] = [json_file_name]
            else:
                json_by_id[img_id].append(json_file_name)

        for img_id in json_by_id:
            single_img_all_runs = json_by_id[img_id]
            all_feature_images = []
            for json_file_name in single_img_all_runs:
                json_file_path = os.path.join(self.runs_dir, json_file_name)
                with open(json_file_path) as json_file:
                    result_data = json.load(json_file)
                    
                    single_run = returnNormalizeMap(result_data['pixel_score'])

                    all_feature_images.append(single_run)
            
            avg_fmap = selected_fmap(all_feature_images)
            avg_fmap = avg_fmap.transpose(2, 0, 1)

            avg_fmap_img = returnNormalizeMap(avg_fmap)
            avg_fmap_img = avg_fmap.transpose(0, 1, 2)

            fv = returnColorFeature(avg_fmap_img)
            fv = fv[...,[2,0,1]].copy() # rgb to bgr
            
            # get ground truth bboxes beside
            img_path = os.path.join(self.img_dir, img_id+'.jpg')
            sample_img = cv2.imread(img_path)
            all_boxes = getBBox(annotation_dir, img_id)

            for bb in all_boxes:
                image = cv2.rectangle(sample_img, (bb['xmin'], bb['ymin']), (bb['xmax'], bb['ymax']), (0, 0, 255), 3)
                
            image_file_path = os.path.join(self.vis_dir, img_id + '_avg.png')
            cv2.imwrite(image_file_path,np.hstack((fv,image)))

        for json_file_name in os.listdir(self.runs_dir):
            if '.json' in json_file_name:
                json_file_path = os.path.join(self.runs_dir, json_file_name)
                os.remove(json_file_path)

    def vis_all_runs(self, img_dir=None, annotation_dir=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir

        for json_file_name in os.listdir(self.runs_dir):
            if '.json' in json_file_name:
                json_file_path = os.path.join(self.runs_dir, json_file_name)
                with open(json_file_path) as json_file:
                    result_data = json.load(json_file)
                    # print (torch.tensor(result_data['pixel_score']).shape)

                    # return feature map
                    fv = returnColorFeature(result_data['pixel_score'])

                    # get ground truth bboxes beside
                    img_id = json_file_name.split('_by')[0]
                    img_path = os.path.join(self.img_dir, img_id+'.jpg')
                    sample_img = cv2.imread(img_path)
                    all_boxes = getBBox(annotation_dir, img_id)

                    for bb in all_boxes:
                        image = cv2.rectangle(sample_img, (bb['xmin'], bb['ymin']), (bb['xmax'], bb['ymax']), (0, 0, 255), 3)


                    image_file_path = os.path.join(self.vis_dir, json_file_name.replace('json','png'))
                    cv2.imwrite(image_file_path,np.hstack((fv,image)))

                    os.remove(json_file_path)

# vis = VisRuns(runs_dir='./FrontBackCore/exp/scale1_num30_models30_2022_07_11_18_13_49/runs')
# vis.vis_average(img_dir= './datasets/full_body/test/objs',
#                 annotation_dir='./datasets/full_body/Annotations')