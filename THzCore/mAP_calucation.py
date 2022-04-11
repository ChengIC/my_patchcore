
import os 
import json
import pandas as pd
import xml.etree.ElementTree as ET
from map_boxes import mean_average_precision_for_boxes

runs_folder='./THzCore/Exp/2022_04_06_13_16_27/runs'
annotation_folder= './datasets/full_body/Annotations'
runs_dict = {
            'ImageID':[],
            'Conf':[],
            'LabelName':[],
            'XMin': [],
            'XMax': [],
            'YMin': [],
            'YMax': [],
}

label_dict = {
            'ImageID':[],
            'LabelName':[],
            'XMin': [],
            'XMax': [],
            'YMin': [],
            'YMax': [],
}


def readXMLpath(xml_filePath, 
            classes = ['GA', 'KK', 'SS' , 'MD', 'CK' , 'WB' , 'KC' , 'CP', 'CL', 'LW', 'UNKNOWN']):
    box_dict = {}
    xml_file = open(xml_filePath, encoding='UTF-8')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    obj_num = 0
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        obj_num +=1
        xmlbox = obj.find('bndbox')

        box_dict[cls] = [
                            int(xmlbox.find('xmin').text),
                            int(xmlbox.find('ymin').text),
                            int(xmlbox.find('xmax').text),
                            int(xmlbox.find('ymax').text)
                    ]
    return box_dict


for run_file in os.listdir(runs_folder):
    if 'overall' in run_file:
        run_file_path = os.path.join(runs_folder,run_file)
        with open(run_file_path) as json_file:
            runs_data = json.load(json_file)
            for bbox in runs_data['picked_boxes']:
                runs_dict['ImageID'].append(runs_data['img_id'])
                runs_dict['Conf'].append(runs_data['filter_score'])
                runs_dict['LabelName'].append('Anomaly')
                runs_dict['XMin'].append(bbox[0])
                runs_dict['XMax'].append(bbox[2])
                runs_dict['YMin'].append(bbox[1])
                runs_dict['YMax'].append(bbox[3])

        # read annotation file
        xml_file_path = os.path.join(annotation_folder,runs_data['img_id']+'.xml')
        box_dict = readXMLpath(xml_file_path)
        for cls in box_dict:
            label_dict['ImageID'].append(runs_data['img_id'])
            label_dict['LabelName'].append('Anomaly')
            label_dict['XMin'].append(box_dict[cls][0])
            label_dict['XMax'].append(box_dict[cls][2])
            label_dict['YMin'].append(box_dict[cls][1])
            label_dict['YMax'].append(box_dict[cls][3])

df1 = pd.DataFrame(runs_dict)
df2= pd.DataFrame(label_dict)

det = df1[['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']].values
ann = df2[['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']].values

mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det, iou_threshold=0.5)
