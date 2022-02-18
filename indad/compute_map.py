import pandas as pd
from map_boxes import mean_average_precision_for_boxes

annotations_csv = '/Users/rc/Documents/GitHub/my_patchcore/processed_result/2022_02_17_12_04_03/data/ground_truth.csv'
detections_csv = '/Users/rc/Documents/GitHub/my_patchcore/processed_result/2022_02_17_12_04_03/data/inference_summary.csv'

ann = pd.read_csv(annotations_csv)
det = pd.read_csv(detections_csv)

ann = ann[['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']].values
det = det[['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']].values
mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det, iou_threshold=0.01)


result_file = 'result.txt'
with open(result_file,'w') as f:
    f.write(str(mean_ap)+'\n')
    f.write(str(average_precisions))

