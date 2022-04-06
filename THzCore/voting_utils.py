

import os
import json
from cores_utils import IoU
from operator import itemgetter
import numpy as np

def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score


class Voting():

    def __init__(self,voting_method='files'):
        
        self.voting_method=voting_method

    
    def getVotes(self,img_results=None,files_list=None):
        bbox_arrs = []
        if self.voting_method=='files' and files_list!=None:
            # generate votes by run files
            # load all boxes
            for json_filePath in files_list:
                json_filename = json_filePath.split('/')[-1]
                im_id = json_filename.split('_config')[0]
                json_id = json_filename.split('_config_')[-1].split('.json')[0]
                
                with open(json_filePath) as json_file:
                    json_data = json.load(json_file)
                    bbox_list =  json_data['detected_box_list']
                    for p in bbox_list:
                        for b in bbox_list[p]:
                            bbox_arrs.append({
                                'p':p,
                                'json_id':json_id,
                                'im_id':im_id,
                                'bbox':b,
                                'score':0
                            })
        else:
            # get inference dictonary without loading
            for exp_info in img_results:
                bbox_list =  exp_info['detected_box_list']
                json_id = exp_info['json_id']
                im_id = exp_info['img_id']

                for p in bbox_list:
                    for b in bbox_list[p]:
                        bbox_arrs.append({
                            'p':p,
                            'json_id':json_id,
                            'im_id':im_id,
                            'bbox':b,
                            'score':0
                        })

        # compute scores
        i = 0
        while i < len(bbox_arrs):
            j = i+1
            while j < len(bbox_arrs):
                if bbox_arrs[i]['json_id']!=bbox_arrs[j]['json_id'] and bbox_arrs[i]['p'] == bbox_arrs[j]['p']:
                    bb1 = bbox_arrs[i]['bbox']
                    bb2 = bbox_arrs[j]['bbox']
                    IoU_Score = IoU(bb1,bb2)
                    if IoU_Score>=0.5:
                        bbox_arrs[i]['score']+=IoU_Score
                        bbox_arrs[j]['score']+=IoU_Score
                else:
                    pass
                j+=1
            i=i+1

        # sort list and filter list
        newlist = sorted(bbox_arrs, key=itemgetter('score'),reverse=True)

        filter_list = []
        filter_score=0.8
        while len(filter_list)==0:
            filter_list = [d for d in newlist if float(d['p'])==filter_score]
            filter_score-=0.05
        chosen_score=filter_score+0.05
        filter_list = [d for d in filter_list if float(d['score'])!=0]
        # print (filter_list)
        
        # add nms on box lists
        max_score = filter_list[0]['score']
        bounding_boxes = [bb['bbox'] for bb in filter_list]
        confidence_score = [bb['score']/max_score for bb in filter_list]
        threshold=0.3
        picked_boxes, picked_score = nms(bounding_boxes, confidence_score, threshold)
        

        # return overall results
        box_results = []
        for (xmin, ymin, xmax, ymax), confidence in zip(picked_boxes, picked_score):
            box_results.append([confidence, xmin, ymin, xmax, ymax])
       
        overall_results = {
            'img_id':bbox_arrs[0]['im_id'],
            'filter_score':chosen_score,
            'picked_boxes':picked_boxes,
            'picked_score':picked_score,
            'box_results':box_results,
        }
        

        # Draw parameters
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 1
        # thickness = 2
        # for (start_x, start_y, end_x, end_y), confidence in zip(picked_boxes, picked_score):
        #     (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
        #     cv2.rectangle(image, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
        #     cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
        #     cv2.putText(image, str(confidence), (start_x, start_y), font, font_scale, (0, 0, 0), thickness)

        return overall_results



if __name__ == "__main__":
    myVote = Voting()

    json_folder = '/Users/rc/Desktop/test_json'
    files_list = [os.path.join(json_folder,f) for f in os.listdir(json_folder)]
    overall_results = myVote.getVotes(files_list=files_list)
    print (overall_results)

