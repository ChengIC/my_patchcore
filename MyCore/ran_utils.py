import os 
import time
import random
import string
import numpy as np
from torch import tensor
import random
import cv2
from skimage.measure import label, regionprops
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image
import io
from matplotlib.pyplot import figure

def getBBox(annotation_dir, img_id):
    xml_filePath = os.path.join (annotation_dir, img_id + '.xml')
    xml_file = open(xml_filePath, encoding='UTF-8')
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls != 'HUMAN':
            xmlbox = obj.find('bndbox')
            xmin = int(xmlbox.find('xmin').text)
            xmax = int(xmlbox.find('xmax').text)
            ymin = int(xmlbox.find('ymin').text)
            ymax = int(xmlbox.find('ymax').text)
            single_box = {
                    'xmin':xmin,
                    'xmax':xmax,
                    'ymin':ymin,
                    'ymax':ymax,
            }
            bboxes.append(single_box)
    return bboxes

IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']
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

def PixelScore2Boxes(pxl_lvl_anom_score):
    anomo_thresholds= [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    score_range = pxl_lvl_anom_score.min(), pxl_lvl_anom_score.max()
    fmap_img = pred_to_img(pxl_lvl_anom_score, score_range)
    detected_box_list = {}
    for anomo_threshold in anomo_thresholds:
        try:
            mask = fmap_img > anomo_threshold
            label_mask = label(mask[:, :, 0])
            props = regionprops(label_mask)
            detected_box_list[str(anomo_threshold)] = []
            for prop in props:
                detected_box =  [int(prop.bbox[1]), int(prop.bbox[0]),
                                int(prop.bbox[3]), int(prop.bbox[2])]  # 1 0 3 2
                detected_box_list[str(anomo_threshold)].append(detected_box)
        except:
            pass

    return detected_box_list

def genTimeStamp():
    now = int(time.time())
    timeArray = time.localtime(now)
    TimeStamp = time.strftime("%Y_%m_%d_%H_%M_%S", timeArray)
    return TimeStamp

def unique_id(size):
    chars = list(set(string.ascii_uppercase + string.digits).difference('LIO01'))
    return ''.join(random.choices(chars, k=size))


def mean_size_folder(training_folder):
    height_list=[]
    width_list=[]
    for img_file in os.listdir(training_folder):
        if img_file.split('.')[-1] in IMG_FORMATS:
            img_path = os.path.join(training_folder,img_file)
            im = cv2.imread(img_path)
            h, w, _ = im.shape
            height_list.append(h)
            width_list.append(w)
    width_list = np.array(width_list)
    height_list = np.array(height_list)
    return [int(np.mean(height_list)),int(np.mean(width_list))]

def renderFeatureMap(pxl_lvl_anom_score,sample_img):
    score_range = pxl_lvl_anom_score.min(), pxl_lvl_anom_score.max()
    fmap_img = pred_to_img(pxl_lvl_anom_score, score_range)
    
    figure(figsize=(10, 10), dpi=200)
    plt.imshow(sample_img)
    plt.imshow(fmap_img, cmap="jet", alpha=0.5)
    plt.axis('off');
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    overlay_img = Image.open(buf)
    plt.clf()
    return np.array(overlay_img), fmap_img