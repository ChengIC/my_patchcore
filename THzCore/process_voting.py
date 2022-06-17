import os
import xml.etree.ElementTree as ET
import pandas as pd
import json
import torch
import torchvision.ops.boxes as bops
from tqdm import tqdm
from map_boxes import mean_average_precision_for_boxes
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def IoU(box1,box2):
    box1 = torch.tensor([box1], dtype=torch.float)
    box2 = torch.tensor([box2], dtype=torch.float)
    iou = bops.box_iou(box1, box2)
    return iou.item()

def add_voting_score(df_example, same_para='bag'):
    df_example_copy = df_example.copy()
    for index, row in df_example.iterrows():
        box1 = [row.XMin, row.XMax, row.YMin, row.YMax]
        voting_score = 0
        voting_score_list = []
        for index2, row2 in df_example_copy.iterrows():
            if row2.config_id!=row.config_id:
                box2 = [row2.XMin, row2.XMax, row2.YMin, row2.YMax]
                if IoU(box1,box2)>0.3:
                    voting_score += 1

            voting_score_list.append(voting_score)
            
        if same_para == 'bag':
            df_example_copy['voting_same_bag_score'] = voting_score_list
        else:
            df_example_copy['voting_same_scale_score'] = voting_score_list
            
    return df_example_copy

def add_voting(studied_id, det_df):
    # new voting scheme
    studied_df = det_df[det_df['ImageID']== studied_id]
    # same bag score
    same_bag_scores = []
    for bag_idx in tqdm(set(studied_df.config_idx)):
        ##print ('processing bag: ' + str(bag_idx))
        same_bag_detection = studied_df [studied_df['config_idx']==bag_idx]
        same_bag_detection = add_voting_score(same_bag_detection,same_para='bag')
        same_bag_scores.append(same_bag_detection)

    df = pd.concat(same_bag_scores, ignore_index=True)
    
    # same scale score 
    same_scale_scores = []
    for scale_idx in tqdm(set(df.config_scale)):
        #print ('processing scale: ' + str(scale_idx))
        same_scale_detection = df [df['config_scale']==scale_idx]
        same_scale_detection = add_voting_score(same_scale_detection, same_para='scale')
        same_scale_scores.append(same_scale_detection)
        
    df = pd.concat(same_scale_scores, ignore_index=True)
    
    image_id = list(set(df.ImageID))[0]
    filepath = os.path.join('/Users/rc/Documents/GitHub/my_patchcore/exp_notebook/new_voting',  image_id +'.csv' )
    
    df.to_csv(filepath,index=False)
    
    return df
    
det_df = pd.read_csv('exp_notebook/runs_075_085.csv')

# continue study
voting_folder = '/Users/rc/Documents/GitHub/my_patchcore/exp_notebook/new_voting'
finised_file_list = os.listdir(voting_folder)
finised_file_list = [f.split('.')[0] for f in finised_file_list if '.csv' in f]
diff_id = set(det_df.ImageID) - set(finised_file_list)

process_frames = Parallel(n_jobs=-1)(delayed(add_voting)(studied_id,det_df) for studied_id in set(diff_id))