
from __future__ import annotations
import os
import cv2
import xml.etree.ElementTree as ET

process_folder_images = './datasets/full_body/test/objs'
annotations_folder = './datasets/full_body/Annotations'
output_folder = './datasets/THz_Body/cropped_objs'
if not os.path.exists(output_folder):
    os.makedirs (output_folder)


for img_filename in os.listdir(process_folder_images):
    img_path = os.path.join(process_folder_images,img_filename)
    if '.jpg' in img_path:
        img = cv2.imread(img_path)
        try:
            xml_filePath=os.path.join(annotations_folder,img_filename.replace('.jpg','.xml'))
            xml_file = open(xml_filePath, encoding='UTF-8')
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls == 'HUMAN':
                    xmlbox = obj.find('bndbox')
                    xmin = int(xmlbox.find('xmin').text)
                    xmax = int(xmlbox.find('xmax').text)
                    ymin = int(xmlbox.find('ymin').text)
                    ymax = int(xmlbox.find('ymax').text)
                    crop_img = img[ymin:ymax, xmin:xmax]
                    output_img_path = os.path.join(output_folder,img_filename)
                    cv2.imwrite(output_img_path, crop_img)
                    continue
        except:
            pass