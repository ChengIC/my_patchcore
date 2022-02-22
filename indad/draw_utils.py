
from PIL import Image
import matplotlib.pyplot as plt
from torch import tensor
import io
from torchvision import transforms
import torch
from skimage.measure import label, regionprops
import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np

IMAGENET_MEAN = tensor([.485, .456, .406])
IMAGENET_STD = tensor([.229, .224, .225]) 

def readXML(annotation_folder, image_name):
    box_dict = {}
    classes = ['GA', 'KK', 'SS' , 'MD', 'CK' , 'WB' , 'KC' , 'CP', 'CL', 'LW', 'UNKNOWN']
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
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text), 
            int(xmlbox.find('ymin').text),  int(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # box_dict[cls] = [b1/335,b2/335,b3/880,b4/880]
        box_dict[cls] = [b1,b2,b3,b4]
    return box_dict

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

def WriteOverlayImage(orinial_image_path,test_img_tensor,
                    img_lvl_anom_score,pxl_lvl_anom_score,output_img_path):
    if test_img_tensor is None:
        image = Image.open(orinial_image_path).convert('RGB')
        loader = load_img()
        image = loader(image).unsqueeze(0)
        test_img_tensor = image.to('cpu', torch.float)

    # generate overlay image
    score_range = pxl_lvl_anom_score.min(), pxl_lvl_anom_score.max()
    # print ('test_img_tensor.shape: ' +str(test_img_tensor.shape))
    # print ('pxl_lvl_anom_score.shape: ' +str(pxl_lvl_anom_score.shape))
    overlay_img = show_pred(test_img_tensor.squeeze(0), img_lvl_anom_score, pxl_lvl_anom_score, score_range)
    
    # Write image
    img = Image.open(orinial_image_path).convert('RGB')
    width1, height1 = img.size
    width2, height2 = overlay_img.size
    if width1!=width2 or height1!=height2:
        overlay_img = overlay_img.resize((width1, height1))

    img2 = Image.new("RGB", (width1+width1, height1), "white")
    img2.paste(img, (0, 0))
    img2.paste(overlay_img, (width1, 0))
    img2.save(output_img_path)


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

    return detected_box_list


def WriteDetectImage(image_path,annotation_folder,detected_box_list,
                            image_name,output_img_path):
    img_1 = cv2.imread(image_path)
    img_2= cv2.imread(image_path)
    for detected_box in detected_box_list:
        cv2.rectangle(img_1, (detected_box[0], detected_box[1]), 
                                (detected_box[2], detected_box[3]), (255,0,0), 2)
    
    box_dict = readXML(annotation_folder, image_name)
    for cls in box_dict:
        bbox = [int(bb) for bb in box_dict[cls]]
        cv2.rectangle(img_2, (bbox[0], bbox[2]), (bbox[1], bbox[3]),  (0, 0, 255), 2)

    numpy_horizontal = np.hstack((img_2, img_1))

    cv2.imwrite(output_img_path, numpy_horizontal)

def combineImages(overlay_img_path,detected_img_path,output_img_path):
    img1 = cv2.imread(overlay_img_path)
    img2 = cv2.imread(detected_img_path)
    im_v = cv2.vconcat([img1, img2])
    cv2.imwrite(output_img_path, im_v)

    os.remove(overlay_img_path)
    os.remove(detected_img_path)


