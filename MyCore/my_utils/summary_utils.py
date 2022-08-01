


import numpy as np
import pandas as pd
import os

class SummaryExp():

    def __init__(self):
        self.data_frame = {
            'img_id':[],
            'tp_score':[],
            'fp_score':[]
        }

    def compute_score (self, fv_img, all_boxes):

        # compute all score
        all_score = np.sum(fv_img)

        # compute tp_score
        tp_score = 0
        for bb in all_boxes:
            crop_img = fv_img[bb['xmin']:bb['xmax'], bb['ymin']:bb['ymax']]
            tp_score += np.sum(crop_img)
        
        # compute fp_score
        fp_score = all_score - tp_score
        
        return tp_score, fp_score

    def update_data_frame (self,img_id, tp_score, fp_score):
        self.data_frame['img_id'].append(img_id)
        self.data_frame['tp_score'].append(tp_score)
        self.data_frame['fp_score'].append(fp_score)

    def write_exp_summary(self, exp_dir):
        summary_pd = pd.DataFrame.from_dict(self.data_frame)
        csv_file_path = os.path.join (exp_dir,'summary.csv' )
        summary_pd.to_csv(csv_file_path, index=False)





