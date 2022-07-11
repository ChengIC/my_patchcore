

import os 
from ran_utils import *
import json

class VisRuns():
    def __init__(self, runs_dir=None, save_name='vis') :
        self.runs_dir = runs_dir

        self.vis_dir = os.path.join ('/'.join(self.runs_dir.split('/')[:-1]), save_name)
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)
    
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

# vis = VisRuns(runs_dir='./FrontBackCore/exp/scale1_num30_models30_2022_07_11_14_05_10/runs')
# vis.vis_all_runs(img_dir= './datasets/full_body/test/objs',
#                 annotation_dir='./datasets/full_body/Annotations')