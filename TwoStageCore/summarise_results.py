

import os 
import pandas as pd
import json
from ran_utils import *
from tqdm import tqdm
def summarise_two_stage(cuts_dir, final_run_dir):
    summary = {
        'img_id':[],
        'crop_id':[],
        'XMin':[],
        'YMin':[],
        'XMax':[],
        'YMax':[],
        'Threshold':[],
        'Score':[],
    }

    for file in tqdm(os.listdir(cuts_dir)):
        if '.json' in file:
            with open(os.path.join(cuts_dir,file)) as run_file:
                run_data = json.load(run_file)
                # print (run_data)

                save_crop_id  = run_data['save_crop_path'].split('.jpg')[0]
                summary['img_id'].append(run_data['img_id'])
                summary['crop_id'].append(save_crop_id)
                summary['XMin'].append(run_data['bbox'][0])
                summary['YMin'].append(run_data['bbox'][1])
                summary['XMax'].append(run_data['bbox'][2])
                summary['YMax'].append(run_data['bbox'][3])
                summary['Threshold'].append(float(run_data['Threshold']))

                target_run_file = None
                for runs_file in os.listdir(final_run_dir):
                    if save_crop_id in runs_file:
                        target_run_file = runs_file
                
                with open(os.path.join(final_run_dir,target_run_file)) as run_file2:
                    run_data2 = json.load(run_file2)
                    summary['Score'].append(run_data2['img_score'])

    summary_data = pd.DataFrame(summary)
    summary_data_path = 'runs' + genTimeStamp() + '.csv'
    summary_data.to_csv(summary_data_path,index=False)
    print('Summary file' + summary_data_path)


cuts_dir = '/Users/rc/Documents/GitHub/my_patchcore/TwoStageCore/exp/2022_07_05_15_15_33/cut_abnormal_patch'
final_run_dir = '/Users/rc/Documents/GitHub/my_patchcore/TwoStageCore/exp/2022_07_05_15_15_33/final_runs'

summarise_two_stage(cuts_dir, final_run_dir)