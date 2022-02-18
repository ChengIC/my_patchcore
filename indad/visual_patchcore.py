
import os
import json
import numpy as np
from save_utils import folderSet
import torch
from draw_utils import WriteOverlayImage,AnomalyToBBox,WriteDetectImage
from draw_utils import readXML
import csv

class visPatchCore():

    def __init__(self, all_json_dirs,test_imgs_folder,annotation_folder,
                write_image=True,write_result=True):

        self.all_json_dirs=all_json_dirs
        
        self.test_imgs_folder=test_imgs_folder
        self.annotation_folder=annotation_folder

        self.write_image=write_image
        self.write_result=write_result
        self.output_img_folder,self.output_data_folder = folderSet()

        self.process_info_file = os.path.join(self.output_data_folder,'info.json')
        data_dict = {
                'all_json_dirs':self.all_json_dirs,
                'test_imgs_folder':self.test_imgs_folder,
                'annotation_folder':self.annotation_folder,
        }
        json_string = json.dumps(data_dict)
        with open(self.process_info_file, 'w') as outfile:
            outfile.write(json_string)

        self.gt_csv_file = os.path.join(self.output_data_folder,'ground_truth.csv')
        gt_csv_header = ['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']
        with open(self.gt_csv_file,'w') as f1:
            writer = csv.writer(f1)
            writer.writerow(gt_csv_header)

        self.det_csv_file = os.path.join(self.output_data_folder,'inference_summary.csv')
        det_csv_header = ['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']
        with open(self.det_csv_file,'w') as f2:
            writer = csv.writer(f2)
            writer.writerow(det_csv_header)

    def vis_result(self):
        
        model_num = len(self.all_json_dirs)
        
        json_ids = [os.listdir(k) for k in self.all_json_dirs]
        mutal_json_ids = set(json_ids[0])
        for s in json_ids[1:]:
            mutal_json_ids.intersection_update(s)
        mutal_ids = list(mutal_json_ids)

        for json_id in mutal_ids:
            accumulated_pixel_scores=None
            accumulated_img_scores=None
            
            for dir in self.all_json_dirs: 
                json_path = os.path.join(dir,json_id)
                with open(json_path) as json_file:
                    json_data = json.load(json_file)
                    pxl_lvl_anom_score = np.array(json_data['pxl_lvl_anom_score'])
                    img_lvl_anom_score = np.array(json_data['image_score'])

                    if accumulated_pixel_scores is None:
                        accumulated_pixel_scores = pxl_lvl_anom_score
                    else:
                        accumulated_pixel_scores = np.add(accumulated_pixel_scores,pxl_lvl_anom_score)
            
                    if accumulated_img_scores is None:
                        accumulated_img_scores = img_lvl_anom_score
                    else:
                        accumulated_img_scores = np.add(accumulated_img_scores,img_lvl_anom_score)
            
            # accumulated result processing
            accumulated_pixel_scores = torch.from_numpy(accumulated_pixel_scores/model_num)
            accumulated_img_scores = torch.as_tensor(accumulated_img_scores/model_num)
            detected_box_list = AnomalyToBBox(accumulated_pixel_scores, anomo_threshold=0.75, x_ratio=1, y_ratio=1)

            if self.write_image:
                image_name =json_id.replace('.json','.jpg')
                image_path = os.path.join(self.test_imgs_folder,image_name)

                overlay_image_name = json_id.replace('.json','_overlay.jpg')
                overlay_img_path = os.path.join(self.output_img_folder,overlay_image_name)
                
                detected_image_name = json_id.replace('.json','_detected.jpg')
                detected_img_path = os.path.join(self.output_img_folder,detected_image_name)
                
                WriteOverlayImage(image_path,None,accumulated_img_scores,
                                    accumulated_pixel_scores,overlay_img_path)
                
                WriteDetectImage(image_path,self.annotation_folder,detected_box_list,
                                image_name,detected_img_path)

            if self.write_result:
                box_dict = readXML(self.annotation_folder, image_name)
                with open(self.gt_csv_file,'a') as f1:
                    writer = csv.writer(f1)
                    for cls in box_dict:
                        bbox = box_dict[cls]
                        csv_data = [image_name, 'Anomaly', bbox[0], bbox[1], bbox[2], bbox[3]]
                        writer.writerow(csv_data)

                with open(self.det_csv_file,'a') as f2:
                    writer = csv.writer(f2)
                    for bbox in detected_box_list:
                        csv_data = [image_name, 'Anomaly' ,0.75, bbox[0], bbox[2], bbox[1], bbox[3]]
                        writer.writerow(csv_data)


if __name__ == "__main__":

    dirs = ['./results/percentage_0.2_XYWQMHA8/2022_02_16_13_57_57/data']
    test_imgs_folder = './datasets/full_body/test/objs'
    annotation_folder = './datasets/full_body/Annotations'
    av1 = visPatchCore (dirs,test_imgs_folder,annotation_folder)
    av1.vis_result()