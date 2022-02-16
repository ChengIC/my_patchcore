import pandas as pd
from map_boxes import mean_average_precision_for_boxes

annotations_csv = './results/multi_core/2022_02_14_13_31_47/csv/ground_truth.csv'
detections_csv = './results/multi_core/2022_02_14_13_31_47/csv/inference_summary.csv'

ann = pd.read_csv(annotations_csv)
det = pd.read_csv(detections_csv)

ann = ann[['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']].values
det = det[['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']].values
mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det, iou_threshold=0.1)


result_file = 'result.txt'
with open(result_file,'w') as f:
    f.write(str(mean_ap)+'\n')
    f.write(str(average_precisions))

