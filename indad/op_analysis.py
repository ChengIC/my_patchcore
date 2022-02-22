import pandas as pd
from map_boxes import mean_average_precision_for_boxes

def bb_intersection_over_union(boxA, boxB):
    # xmin,ymin,xmax,ymax
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = (xB - xA + 1) * (yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


annotations_csv = 'processed_results/2022_02_22_11_11_28/results_example/data/ground_truth.csv'
detections_csv = 'processed_results/2022_02_22_11_11_28/results_example/data/inference_summary.csv'

ann = pd.read_csv(annotations_csv)
det = pd.read_csv(detections_csv)

ann = ann[['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']].values
det = det[['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']].values

filter_det = []
for a in ann:
    for d in det:
        if d[0] == a[0]:
            boxA= [a[2],a[4],a[3],a[5]]
            boxB= [d[3],d[5],d[4],d[6]]
            if bb_intersection_over_union(boxA,boxB)>0:
                filter_det.append(d)

mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det, iou_threshold=0.1)

mean_ap, average_precisions = mean_average_precision_for_boxes(ann, filter_det, iou_threshold=0.1)


# result_file = 'result.txt'
# with open(result_file,'w') as f:
#     f.write(str(mean_ap)+'\n')
#     f.write(str(average_precisions))

