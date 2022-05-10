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
import json
from torchvision import transforms
from torch import tensor
import matplotlib.pyplot as plt
import io

random.seed(20220308)


def load_img():
    loader=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
    return loader

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

def show_pred(sample, score, fmap, range):
    sample_img = tensor_to_img(sample, normalize=True)
    fmap_img = pred_to_img(fmap, range)
    # overlay
    plt.imshow(sample_img)
    plt.imshow(fmap_img, cmap="jet", alpha=0.5)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    # buf.seek(0)
    overlay_img = Image.open(buf)

    return overlay_img


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

def WriteDetectImage(input_image_path,annotation_folder,detected_box_list,
                            image_name,output_img_path):
    img_1 = cv2.imread(input_image_path)
    img_2= cv2.imread(input_image_path)
    for detected_box in detected_box_list:
        cv2.rectangle(img_1, (detected_box[0], detected_box[1]), 
                                (detected_box[2], detected_box[3]), (255,0,0), 2)
    
    box_dict = readXML(annotation_folder, image_name)
    for cls in box_dict:
        bbox = [int(bb) for bb in box_dict[cls]]
        cv2.rectangle(img_2, (bbox[0], bbox[1]), (bbox[2], bbox[3]),  (0, 0, 255), 2)

    numpy_horizontal = np.hstack((img_2, img_1))

    cv2.imwrite(output_img_path, numpy_horizontal)


def load_model_from_dir(model, dir):
	state_dict_path = os.path.join(dir,'Core_Path')
	tar_path = os.path.join(dir,'Core_Path.tar')
	configPath = os.path.join(dir,'loader_info.json')

	if torch.cuda.is_available():
		model.load_state_dict(torch.load(state_dict_path))
		model_paras = torch.load(tar_path)
	else:
		model.load_state_dict(torch.load(state_dict_path,map_location ='cpu'))
		model_paras = torch.load(tar_path,map_location ='cpu')
	
	with open(configPath) as json_file:
			data = json.load(json_file)
			resize_box=data['box']

	return model, model_paras, resize_box

def loader_from_resize(resize_box):
	IMAGENET_MEAN = tensor([.485, .456, .406])
	IMAGENET_STD = tensor([.229, .224, .225])
	transfoms_paras = [
			transforms.Resize(resize_box, interpolation=transforms.InterpolationMode.BICUBIC),
			transforms.ToTensor(),
			transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
	]
	return transforms.Compose(transfoms_paras)



def visOverlay(img_path,pxl_lvl_anom_score):
    sample_img = Image.open(img_path).convert('RGB')

    score_range = pxl_lvl_anom_score.min(), pxl_lvl_anom_score.max()
    fmap_img = pred_to_img(pxl_lvl_anom_score, score_range)
    
    # overlay
    plt.imshow(sample_img)
    plt.imshow(fmap_img, cmap="jet", alpha=0.5)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    # buf.seek(0)
    overlay_img = Image.open(buf)

    width1, height1 = sample_img.size
    width2, height2 = overlay_img.size
    if width1!=width2 or height1!=height2:
        overlay_img = overlay_img.resize((width1, height1))
    
    img2 = Image.new("RGB", (width1+width1, height1), "white")
    img2.paste(sample_img, (0, 0))
    img2.paste(overlay_img, (width1, 0))
    buf.close()

    return img2

def PixelScore2Boxes(pxl_lvl_anom_score):
    anomo_thresholds= [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    score_range = pxl_lvl_anom_score.min(), pxl_lvl_anom_score.max()
    fmap_img = pred_to_img(pxl_lvl_anom_score, score_range)
    detected_box_list = {}
    for anomo_threshold in anomo_thresholds:
        mask = fmap_img > anomo_threshold
        label_mask = label(mask[:, :, 0])
        props = regionprops(label_mask)
        detected_box_list[str(anomo_threshold)] = []
        for prop in props:
            detected_box =  [int(prop.bbox[1]), int(prop.bbox[0]),
                            int(prop.bbox[3]), int(prop.bbox[2])]  # 1 0 3 2
            detected_box_list[str(anomo_threshold)].append(detected_box)

    return detected_box_list


def BoxesfromJson(jsonPath,output_path,annotation_dir=None):
    
    with open(jsonPath) as json_file:
        data = json.load(json_file)
        box_dict =  data['detected_box_list']
        img = cv2.imread(data['img_path'])
        gt_img = img.copy()
        for threshold in box_dict:
            bboxes = box_dict[threshold]
            color = list(np.random.random(size=3) * 256)
            for bbox in bboxes:
                img = cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color, 2)
    
    if annotation_dir!=None:
        image_name = data['img_path'].split('/')[-1]
        gt_boxList = readXML(annotation_dir, image_name)

        for cls in gt_boxList:
            bbox = gt_boxList[cls]
            gt_img = cv2.rectangle(gt_img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255,0,0), 2)
    
    vis = np.concatenate((gt_img, img), axis=1)

    cv2.imwrite(output_path,vis)


