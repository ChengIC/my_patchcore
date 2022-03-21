import os 
import time
import torch
import random
from PIL import Image
from torch import tensor
from skimage.measure import label, regionprops
import numpy as np 
import xml.etree.ElementTree as ET
import torchvision.ops.boxes as bops
import cv2
import string

random.seed(20220308)

def unique_id(size):
    chars = list(set(string.ascii_uppercase + string.digits).difference('LIO01'))
    return ''.join(random.choices(chars, k=size))

def genTimeStamp():
    now = int(time.time())
    timeArray = time.localtime(now)
    TimeStamp = time.strftime("%Y_%m_%d_%H_%M_%S", timeArray)
    return TimeStamp

def SplitList(dir,percentage=0.2):
	img_list = os.listdir(dir)
	random.shuffle(img_list)
	len_list = int(len(os.listdir(dir))*percentage)
	my_SplitList=[]
	for idx in range(0,len_list):
		im_id = img_list[idx]
		if '.jpg' in im_id:
			my_SplitList.append(os.path.join(dir,im_id))
	return my_SplitList

IMAGENET_MEAN = tensor([.485, .456, .406])
IMAGENET_STD = tensor([.229, .224, .225]) 

def tensor_to_img(x, normalize=False):
    if normalize:
        x *= IMAGENET_STD.unsqueeze(-1).unsqueeze(-1)
        x += IMAGENET_MEAN.unsqueeze(-1).unsqueeze(-1)
    x =  x.clip(0.,1.).permute(1,2,0).detach().numpy()
    return x

def pred_to_img(x, range): 
    range_min, range_max = range
    x -= range_min
    if (range_max - range_min) > 0:
        x /= (range_max - range_min)
    return tensor_to_img(x)

def AnomalyToBBox(pxl_lvl_anom_score, anomo_threshold=0.75, x_ratio=1, y_ratio=1):
    score_range = pxl_lvl_anom_score.min(), pxl_lvl_anom_score.max()
    fmap_img = pred_to_img(pxl_lvl_anom_score, score_range)    
    mask = fmap_img > anomo_threshold
    label_mask = label(mask[:, :, 0])
    props = regionprops(label_mask)
    detected_box_list = []
    for prop in props:
        detected_box =  [int(x_ratio * prop.bbox[1]), int(y_ratio * prop.bbox[0]),
                        int(x_ratio * prop.bbox[3]), int(y_ratio * prop.bbox[2])]  # 1 0 3 2
        detected_box_list.append(detected_box)
    detected_box_list = np.array(detected_box_list)
    return detected_box_list

def Image2AnomoBox(imPath,PatchCore_loader,PacthCore_model,PacthCore_model_tar,anomo_threshold=0.5):
    image = Image.open(imPath).convert('RGB')
    original_size_width, original_size_height = image.size
    image = PatchCore_loader(image).unsqueeze(0)
    test_img_tensor = image.to('cpu', torch.float)
    # print (test_img_tensor.shape)
    HeatMap_Size = [original_size_height, original_size_width]
    _, pxl_lvl_anom_score = PacthCore_model.inference(test_img_tensor,PacthCore_model_tar,HeatMap_Size)
    detected_box_list = AnomalyToBBox(pxl_lvl_anom_score, anomo_threshold=anomo_threshold)
    return detected_box_list

def readXML(annotation_folder, image_name, 
            classes = ['GA', 'KK', 'SS' , 'MD', 'CK' , 'WB' , 'KC' , 'CP', 'CL', 'LW', 'UNKNOWN']):
    box_dict = {}
    xml_filePath = os.path.join(annotation_folder ,image_name.split('.jpg')[0] + '.xml')
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

def IoU(box1,box2):
    box1 = torch.tensor([box1], dtype=torch.float)
    box2 = torch.tensor([box2], dtype=torch.float)
    iou = bops.box_iou(box1, box2)
    return iou.item()

def savePatchImg(input_img_path,bbox,output_img_dir):
    img = cv2.imread(input_img_path)
    xmin,ymin,xmax,ymax=bbox
    crop_img = img[ymin:ymax,xmin:xmax]
    output_img_path = os.path.join(output_img_dir,unique_id(18)+'.jpg')
    cv2.imwrite(output_img_path,crop_img)

    return output_img_path

    